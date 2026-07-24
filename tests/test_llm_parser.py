from src import llm_parser


def test_arm_direction_words_override_inverted_small_model_output(monkeypatch):
    response = {
        "choices": [{
            "message": {
                "content": (
                    '{"breed":null,"zone":null,"actions":[],'
                    '"distance_cm":null,"turn_deg":null,'
                    '"manual_key":"r","manual_action":"down"}'
                )
            }
        }]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            import json
            return json.dumps(response).encode()

    monkeypatch.setattr(llm_parser.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    result = llm_parser.parse_with_llm("把机器人的手臂往上抬")
    assert result["manual_key"] == "f"
