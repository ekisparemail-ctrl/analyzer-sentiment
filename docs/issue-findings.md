.venv/Scripts/python src/main.py
Traceback (most recent call last):
  File "C:\Users\kacang\IdeaProjects\analyzer_sentiment\src\main.py", line 103, in <module>
    main()
    ~~~~^^
  File "C:\Users\kacang\IdeaProjects\analyzer_sentiment\src\main.py", line 80, in main
    settings = Settings()  # type: ignore[call-arg]
  File "C:\Users\kacang\IdeaProjects\analyzer_sentiment\.venv\Lib\site-packages\pydantic_settings\main.py", line 262, in __init__
    super().__init__(**__pydantic_self__.__class__._settings_build_values(sources, init_kwargs))
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\kacang\IdeaProjects\analyzer_sentiment\.venv\Lib\site-packages\pydantic\main.py", line 263, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
pydantic_core._pydantic_core.ValidationError: 3 validation errors for Settings
kafka_bootstrap_servers
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
llm_base_url
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
llm_model
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/missing
