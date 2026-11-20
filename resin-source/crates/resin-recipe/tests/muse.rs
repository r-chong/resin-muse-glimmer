use resin_recipe::{
    Event, FinishReason, MuseGlimmerParser, MuseGlimmerTokens, Recipe, muse::MuseGlimmerRecipe,
};
use serde_json::{Value, json};
mod common;
use common::{ByteDecoder, SemanticParser, bytes, coalesce};

const TOKENS: MuseGlimmerTokens = MuseGlimmerTokens {
    start: 1000,
    message: 1001,
    eom: 1002,
    eot: 1003,
    end_of_text: 1004,
};

fn header(recipient: &str) -> Vec<u32> {
    let mut ids = vec![TOKENS.start];
    ids.extend(bytes(&format!("assistant to={recipient}")));
    ids.push(TOKENS.message);
    ids
}

#[test]
fn pinned_hf_template_matches_system_reasoning_tools_and_media_slots() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/muse_hf.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = MuseGlimmerRecipe::default()
            .render(&case["request"])
            .unwrap();
        let mut text = actual.rendered_text();
        // HF's template emits one `<|patch|>` per image; the processor adds the
        // image start/end markers around its expansion, which the recipe renders.
        text = text.replace("<|image_start|><|patch|><|image_end|>", "<|patch|>");
        assert_eq!(text, case["text"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn unknown_options_and_video_fail() {
    let recipe = MuseGlimmerRecipe::default();
    let messages = json!([{"role": "user", "content": "hi"}]);
    assert!(
        recipe
            .render(&json!({"messages": messages, "extra": {"temperature_hint": 1}}))
            .is_err()
    );
    assert!(
        recipe
            .render(&json!({"messages": [{"role": "user", "content": [{"type": "video"}]}]}))
            .is_err()
    );
}

#[test]
fn reasoning_answer_and_tool_calls_are_independent_of_chunk_boundaries() {
    // The generation prompt ends with an open `<|start|>assistant` header.
    let prompt = {
        let mut ids = vec![TOKENS.start];
        ids.extend(bytes("assistant"));
        ids
    };
    let mut ids = bytes(" to=self");
    ids.push(TOKENS.message);
    ids.extend(bytes("Think 🦀."));
    ids.push(TOKENS.eom);
    ids.extend(header("weather.get"));
    ids.extend(bytes(
        "<atem:function_calls>\n<atem:invoke name=\"weather.get\">\n<atem:parameter name=\"city\">東京</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
    ));
    ids.push(TOKENS.eom);
    ids.extend(header("user"));
    ids.extend(bytes("Sunny. <|eot|> stays literal."));
    ids.push(TOKENS.eot);
    ids.extend(bytes("ignored"));
    let expected = vec![
        Event::ReasoningDelta {
            delta: "Think 🦀.".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "weather.get".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"city":"東京","days":3}"#.into(),
        },
        Event::TextDelta {
            delta: "Sunny. <|eot|> stays literal.".into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser =
            MuseGlimmerParser::new(TOKENS, Box::<ByteDecoder>::default(), &prompt).unwrap();
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "chunk width {width}");
    }
}

#[test]
fn headerless_message_and_truncated_tool_call() {
    let mut prompt = vec![TOKENS.start];
    prompt.extend(bytes("assistant"));
    let mut parser =
        MuseGlimmerParser::new(TOKENS, Box::<ByteDecoder>::default(), &prompt).unwrap();
    let mut ids = vec![TOKENS.message];
    ids.extend(bytes("Direct."));
    ids.push(TOKENS.eom);
    ids.extend(header("search"));
    ids.extend(bytes(
        "<atem:function_calls>\n<atem:invoke name=\"search\">",
    ));
    let events = parser.semantic_feed(&ids).unwrap();
    assert_eq!(
        coalesce(events),
        vec![Event::TextDelta {
            delta: "Direct.".into()
        }]
    );
    assert!(
        parser
            .semantic_finish(FinishReason::Length)
            .unwrap()
            .is_empty()
    );
}
