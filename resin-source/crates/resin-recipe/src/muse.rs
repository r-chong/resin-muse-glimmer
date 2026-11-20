//! Muse Glimmer's ATEM conversation format.
//! Source: meta-models/Muse-Glimmer-30B@cec94fcfd6512f2cfe708138c32533b346ea4f69
//! `chat_template.jinja` and `tokenizer_config.json` `response_template`.
//!
//! Messages are `<|start|>{role}[ to={recipient}]<|message|>{body}<|eot|>`.
//! Assistant reasoning is addressed `to=self` and closed by `<|eom|>`; the
//! answer is addressed `to=user`; tool calls are addressed to the function and
//! carry an `<atem:function_calls>` XML body.

use serde_json::{Map, Value};

use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};
use crate::render::{function, list, messages};
use crate::{AttributedText, Decoder, Error, FinishReason, Prompt, ResponseParser};

const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant.";
const DEFAULT_CUTOFF: &str = "2026-01-04";
const TOOL_PREAMBLE: &str = concat!(
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n",
    "You can invoke a function by writing a \"<atem:function_calls>\" block like the following:\n",
    "<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n",
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\n",
    "Here are the functions available in JSONSchema format:\n",
    "// Tool metadata\n",
);
const TOOL_EXAMPLE: &str = concat!(
    "\n\nHere's an example of how to call a function in the tool set:\n",
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than `example_tool_name.example_function_name`)\n\n",
    "to=example_tool_name.example_function_name\n\n",
    "<atem:function_calls>\n<atem:invoke name=\"example_tool_name.example_function_name\">\n",
    "<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n",
    "<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n",
    "</atem:invoke>\n</atem:function_calls>",
);

/// Rendering defaults. `current_date` mirrors the template's `strftime_now`
/// line; the Python factory fills today's date when it is omitted, as
/// Transformers does. Pass it explicitly for reproducible prompts.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MuseGlimmerRecipeParams {
    pub add_generation_prompt: Option<bool>,
    pub reasoning_strength: Option<String>,
    /// OpenAI-style alias for `reasoning_strength`.
    pub reasoning_effort: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub current_date: Option<String>,
    pub tool_namespace_descriptions: Option<Map<String, Value>>,
    #[serde(skip_serializing)]
    pub media_config: Option<crate::media::muse::Config>,
}

#[derive(Clone, Debug, Default)]
pub struct MuseGlimmerRecipe {
    params: MuseGlimmerRecipeParams,
}

impl MuseGlimmerRecipe {
    pub fn new(params: MuseGlimmerRecipeParams) -> Self {
        Self { params }
    }
}

fn json(value: &Value) -> String {
    crate::json::spaced(value)
}

/// `render_content`: text verbatim, one image slot per image part.
fn content(prompt: &mut Prompt, value: Option<&Value>) -> Result<(), Error> {
    match value {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => {
            prompt.text(text);
        }
        Some(Value::Array(parts)) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        prompt.text(
                            part.get("text")
                                .and_then(Value::as_str)
                                .ok_or_else(|| Error::new("Muse text content must be a string"))?,
                        );
                    }
                    // The processor expands each `<|patch|>` slot to
                    // `<|image_start|>` + N `<|patch|>` + `<|image_end|>`.
                    Some("image" | "image_url") => {
                        prompt
                            .special("<|image_start|>")
                            .special("<|patch|>")
                            .special("<|image_end|>");
                    }
                    Some("video" | "video_url") => {
                        return Err(Error::new("Muse video input is not supported"));
                    }
                    _ => {}
                }
            }
        }
        Some(_) => return Err(Error::new("Muse content must be text or content blocks")),
    }
    Ok(())
}

fn plain(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

fn namespaces(tools: &[Value]) -> Vec<String> {
    let mut seen = Vec::<String>::new();
    for tool in tools {
        let name = function(tool)
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let namespace = name.split('.').next().unwrap_or("").to_owned();
        if !seen.contains(&namespace) {
            seen.push(namespace);
        }
    }
    seen
}

fn tool_definitions(tools: &[Value], descriptions: Option<&Map<String, Value>>) -> String {
    let mut out = TOOL_PREAMBLE.to_owned();
    for namespace in namespaces(tools) {
        let description = descriptions
            .and_then(|map| map.get(&namespace))
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()));
        out.push_str(&format!(
            "{{\"name\": {}, \"description\": {}}}\n",
            json(&Value::String(namespace)),
            json(&description)
        ));
    }
    out.push_str("// Function schemas");
    for tool in tools {
        let tool = function(tool);
        let field = |key| tool.get(key).cloned().unwrap_or(Value::Null);
        out.push_str(&format!(
            "\n{{\"name\": {}, \"description\": {}, \"parameters\": {}}}",
            json(&field("name")),
            json(&field("description")),
            json(&field("parameters"))
        ));
    }
    out.push_str(TOOL_EXAMPLE);
    out
}

fn system_meta(tools: &[Value]) -> String {
    let mut recipients = vec!["\"self\"".to_owned()];
    if !tools.is_empty() {
        recipients.extend(
            namespaces(tools)
                .into_iter()
                .map(|ns| format!("\"{ns}.*\"")),
        );
    }
    recipients.push("\"user\"".into());
    format!("# Valid recipients: {}.", recipients.join(", "))
}

fn atem_call(call: &Value) -> Result<String, Error> {
    let call = function(call);
    let name = call.get("name").and_then(Value::as_str).unwrap_or("");
    let arguments = match call.get("arguments") {
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(Value::String(text)) => serde_json::from_str::<Map<String, Value>>(text)
            .map_err(|_| Error::new("Muse tool call arguments must be a JSON object"))?,
        None | Some(Value::Null) => Map::new(),
        _ => return Err(Error::new("Muse tool call arguments must be a JSON object")),
    };
    let mut out = format!("<atem:function_calls>\n<atem:invoke name=\"{name}\">\n");
    for (key, value) in &arguments {
        let value = match value {
            Value::Bool(true) => "true".to_owned(),
            Value::Bool(false) => "false".to_owned(),
            Value::Null => "null".to_owned(),
            Value::String(text) => text.clone(),
            Value::Object(_) | Value::Array(_) | Value::Number(_) => json(value),
        };
        out.push_str(&format!(
            "<atem:parameter name=\"{key}\">{value}</atem:parameter>\n"
        ));
    }
    out.push_str("</atem:invoke>\n</atem:function_calls>");
    Ok(out)
}

/// The result's `name`, else the function whose call ID matches (the last
/// match wins, as in the template), else the bare call ID.
fn tool_result_name(message: &Value, messages: &[Value]) -> String {
    if let Some(name) = message
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
    {
        return name.to_owned();
    }
    let id = message.get("tool_call_id").and_then(Value::as_str);
    let mut name = id.unwrap_or("").to_owned();
    if let Some(id) = id {
        for call in messages.iter().flat_map(|m| list(&m["tool_calls"])) {
            if call.get("id").and_then(Value::as_str) == Some(id) {
                name = function(call)
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
            }
        }
    }
    name
}

fn normalize_reasoning_directive(text: &str) -> String {
    text.replace("Reasoning effort", "Reasoning strength")
        .replace("Reasoning Effort", "Reasoning Strength")
        .replace("reasoning effort", "reasoning strength")
        .replace("REASONING EFFORT", "REASONING STRENGTH")
}

impl crate::Recipe for MuseGlimmerRecipe {
    fn media_config(&self) -> Option<crate::MediaConfig> {
        Some(crate::MediaConfig::Muse(
            self.params.media_config.clone().unwrap_or_default(),
        ))
    }
    fn message_policy(&self) -> crate::MessagePolicy {
        crate::MessagePolicy {
            roles: &["system", "user", "assistant", "tool"],
            ..crate::MessagePolicy::STANDARD
        }
    }
    fn render(&self, request: &Value) -> Result<Prompt, Error> {
        let options = crate::recipe::options("muse_glimmer", &self.params, request)?;
        let messages = messages(request, "Muse Glimmer")?;
        let tools = list(&request["tools"]);
        let strength = options
            .reasoning_strength
            .as_deref()
            .or(options.reasoning_effort.as_deref())
            .filter(|value| !value.is_empty())
            .unwrap_or("high");
        let reasoning = format!("Reasoning strength: {strength}.");
        let descriptions = options.tool_namespace_descriptions.as_ref();
        fn system_tail(
            prompt: &mut Prompt,
            tools: &[Value],
            descriptions: Option<&Map<String, Value>>,
        ) {
            if !tools.is_empty() {
                prompt.text("\n\n");
                prompt.text(tool_definitions(tools, descriptions));
            }
            prompt.text("\n\n");
            prompt.text(system_meta(tools));
            prompt.special("<|eot|>");
        }
        fn role(message: &Value) -> &str {
            message.get("role").and_then(Value::as_str).unwrap_or("")
        }

        let mut prompt = Prompt::new();
        prompt.special("<|begin_of_text|>");
        if !messages.iter().any(|message| role(message) == "system") {
            prompt.special("<|start|>");
            prompt.text("system");
            prompt.special("<|message|>");
            let cutoff = options
                .knowledge_cutoff
                .as_deref()
                .filter(|value| !value.is_empty())
                .unwrap_or(DEFAULT_CUTOFF);
            let mut text = format!("{DEFAULT_SYSTEM}\nKnowledge cutoff: {cutoff}.");
            if let Some(date) = options.current_date.as_deref().filter(|d| !d.is_empty()) {
                text.push_str(&format!("\nCurrent date: {date}."));
            }
            text.push_str("\n\n");
            text.push_str(&reasoning);
            prompt.text(text);
            system_tail(&mut prompt, tools, descriptions);
        }
        for (index, message) in messages.iter().enumerate() {
            let this_role = role(message);
            let end_token = if messages
                .get(index + 1)
                .is_some_and(|next| role(next) == this_role)
            {
                "<|eom|>"
            } else {
                "<|eot|>"
            };
            match this_role {
                "system" => {
                    let text = normalize_reasoning_directive(&plain(message.get("content")));
                    prompt.special("<|start|>");
                    prompt.text("system");
                    prompt.special("<|message|>");
                    prompt.text(&text);
                    if !text.to_lowercase().contains("reasoning strength") {
                        prompt.text("\n\n");
                        prompt.text(&reasoning);
                    }
                    system_tail(&mut prompt, tools, descriptions);
                }
                "user" => {
                    prompt.special("<|start|>");
                    prompt.text("user");
                    prompt.special("<|message|>");
                    content(&mut prompt, message.get("content"))?;
                    prompt.special("<|eot|>");
                }
                "tool" => {
                    let name = tool_result_name(message, messages);
                    prompt.special("<|start|>");
                    prompt.text(format!("tool {name}"));
                    prompt.special("<|message|>");
                    prompt.text(format!("<tool_output name=\"{name}\">\n"));
                    content(&mut prompt, message.get("content"))?;
                    prompt.text("\n</tool_output>");
                    prompt.special("<|eot|>");
                }
                "assistant" => {
                    if let Some(reasoning) = message
                        .get("reasoning_content")
                        .and_then(Value::as_str)
                        .filter(|text| !text.is_empty())
                    {
                        prompt.special("<|start|>");
                        prompt.text("assistant to=self");
                        prompt.special("<|message|>");
                        prompt.text(reasoning);
                        prompt.special("<|eom|>");
                    }
                    let calls = list(&message["tool_calls"]);
                    if !calls.is_empty() {
                        for (call_index, call) in calls.iter().enumerate() {
                            let name = function(call)
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            prompt.special("<|start|>");
                            prompt.text(format!("assistant to={name}"));
                            prompt.special("<|message|>");
                            prompt.text(atem_call(call)?);
                            prompt.special(if call_index + 1 == calls.len() {
                                end_token
                            } else {
                                "<|eom|>"
                            });
                        }
                    } else {
                        let recipient = message
                            .get("recipient")
                            .and_then(Value::as_str)
                            .filter(|r| !r.is_empty())
                            .unwrap_or("user");
                        let end_turn = message
                            .get("end_turn")
                            .and_then(Value::as_bool)
                            .unwrap_or(recipient == "user");
                        prompt.special("<|start|>");
                        prompt.text(format!("assistant to={recipient}"));
                        prompt.special("<|message|>");
                        content(&mut prompt, message.get("content"))?;
                        prompt.special(if end_turn { "<|eot|>" } else { "<|eom|>" });
                    }
                }
                other => {
                    return Err(Error::new(format!("Muse does not support role {other:?}")));
                }
            }
        }
        if options.add_generation_prompt.unwrap_or(true) {
            prompt.special("<|start|>");
            prompt.text("assistant");
        }
        Ok(prompt)
    }
    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn Decoder>,
        _request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, Error> {
        Ok(Box::new(MuseGlimmerParser::new(
            MuseGlimmerTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, Error> {
        Ok(vec![
            crate::parser::resolve(encoder, &["<|eot|>"])?,
            crate::parser::resolve(encoder, &["<|end_of_text|>"])?,
        ])
    }
}

tokens! { MuseGlimmerTokens {
    start => ["<|start|>"],
    message => ["<|message|>"],
    eom => ["<|eom|>"],
    eot => ["<|eot|>"],
    end_of_text => ["<|end_of_text|>"],
}}

enum State {
    Header,
    Text,
    Reasoning,
    Tool,
    Ignore,
}

/// ATEM headers and bodies. A tool call commits when its message closes.
pub type MuseGlimmerParser = Parser<MuseGlimmerGrammar>;
pub struct MuseGlimmerGrammar {
    tokens: MuseGlimmerTokens,
    state: State,
    header: String,
    body: String,
    tool_index: usize,
    ended: bool,
}

impl MuseGlimmerParser {
    pub fn new(
        tokens: MuseGlimmerTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let mut parser = Self::with_grammar(
            MuseGlimmerGrammar {
                tokens,
                state: State::Text,
                header: String::new(),
                body: String::new(),
                tool_index: 0,
                ended: false,
            },
            decoder,
        );
        // Seed only the generation prompt's open header, never earlier turns.
        if let Some(start) = prompt_ids.iter().rposition(|&id| id == tokens.start) {
            let tail = &prompt_ids[start..];
            let end = tail.iter().position(|&id| id == tokens.message);
            parser.feed(&tail[..end.map_or(tail.len(), |end| end + 1)])?;
        }
        parser.next_token_index = 0;
        Ok(parser)
    }
}

/// Parse `<atem:invoke name="…">` blocks into (name, JSON-object arguments).
/// Parameter values that parse as JSON keep that type; others are strings.
pub fn parse_atem_calls(body: &str) -> Vec<(String, String)> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(open) = rest.find("<atem:invoke") {
        rest = &rest[open..];
        let Some(name_at) = rest.find("name=\"") else {
            break;
        };
        let after_name = &rest[name_at + 6..];
        let Some(name_end) = after_name.find('"') else {
            break;
        };
        let name = after_name[..name_end].to_owned();
        let Some(tag_end) = after_name[name_end..].find('>') else {
            break;
        };
        let inner_start = &after_name[name_end + tag_end + 1..];
        let Some(close) = inner_start.find("</atem:invoke>") else {
            break;
        };
        let inner = &inner_start[..close];
        let mut arguments = Map::new();
        let mut params = inner;
        while let Some(p) = params.find("<atem:parameter") {
            params = &params[p..];
            let Some(key_at) = params.find("name=\"") else {
                break;
            };
            let after_key = &params[key_at + 6..];
            let Some(key_end) = after_key.find('"') else {
                break;
            };
            let key = after_key[..key_end].to_owned();
            let Some(gt) = after_key[key_end..].find('>') else {
                break;
            };
            let value_start = &after_key[key_end + gt + 1..];
            let Some(value_end) = value_start.find("</atem:parameter>") else {
                break;
            };
            let raw = &value_start[..value_end];
            let value = serde_json::from_str::<Value>(raw)
                .unwrap_or_else(|_| Value::String(raw.to_owned()));
            arguments.insert(key, value);
            params = &value_start[value_end + "</atem:parameter>".len()..];
        }
        calls.push((name, Value::Object(arguments).to_string()));
        rest = &inner_start[close + "</atem:invoke>".len()..];
    }
    calls
}

impl MuseGlimmerGrammar {
    fn open_body(&mut self) {
        let mut words = self.header.split_whitespace();
        let role = words.next().unwrap_or("");
        let recipient = self
            .header
            .split_whitespace()
            .find_map(|word| word.strip_prefix("to="));
        self.body.clear();
        self.state = if role != "assistant" {
            State::Ignore
        } else {
            match recipient {
                Some("self") => State::Reasoning,
                None | Some("user") | Some("all") => State::Text,
                Some(_) => State::Tool,
            }
        };
    }
    fn close_body(&mut self, events: &mut EventSink) {
        if matches!(self.state, State::Tool) {
            for (name, arguments) in parse_atem_calls(&self.body) {
                emit_tool(&mut self.tool_index, name, arguments, events);
            }
        }
        self.body.clear();
        self.state = State::Ignore;
    }
}

impl Grammar for MuseGlimmerGrammar {
    fn channel(&self) -> Option<&'static str> {
        match self.state {
            State::Text => Some("content"),
            State::Reasoning => Some("reasoning"),
            State::Tool => Some("tool"),
            _ => None,
        }
    }
    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if !text.is_empty() {
            match self.state {
                State::Header => self.header.push_str(&text),
                State::Text => events.text(text, false),
                State::Reasoning => events.text(text, true),
                State::Tool => self.body.push_str(&text),
                State::Ignore => {}
            }
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        if id == self.tokens.start {
            self.body.clear();
            self.header.clear();
            self.state = State::Header;
        } else if id == self.tokens.message && matches!(self.state, State::Header) {
            self.open_body();
        } else if id == self.tokens.eom {
            self.close_body(events);
        } else if id == self.tokens.eot || id == self.tokens.end_of_text {
            self.close_body(events);
            self.ended = true;
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, _events: &mut EventSink) -> Result<(), Error> {
        self.body.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atem_calls_keep_json_types_and_raw_strings() {
        let body = "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">東京</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n<atem:parameter name=\"opts\">{\"unit\": \"c\"}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        assert_eq!(
            parse_atem_calls(body),
            vec![(
                "weather.get".to_owned(),
                r#"{"city":"東京","days":3,"opts":{"unit":"c"}}"#.to_owned()
            )]
        );
        assert!(parse_atem_calls("<atem:invoke name=\"x\">unterminated").is_empty());
    }
}
