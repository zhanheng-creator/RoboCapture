# Task generation and task planning

## Related documents

```shell
command/
│  main_new_command1_dismantle.py
│  main_new_command1_generate_fenlei.py
│  main_new_command1_generate_tuili.py
│  main_new_command1_generate_zhengli.py
│  main_new_command2_dismantle.py
│  main_new_command2_generate.py
│  main_new_command3_generate.py
│
├─data_my
│      command2_generate_examples.xlsx
│      command3_generate_examples.xlsx
│
├─data_new
└─data_to_json_9_8
        command_four_data_to_json.py
        command_one_data_to_json.py
        command_one_to_three_dismantle_data_to_json.py
        command_three_data_to_json.py
        command_two_data_to_json.py
        command_two_to_three_dismantle_data_to_json.py
```

- main_new_command1_dismantle.py：One-level instruction task planning code

- main_new_command1_generate_fenlei.py：One-level instruction (classification) generation code

- main_new_command1_generate_tuili.py：One-level instruction (inference) generation code

- main_new_command1_generate_zhengli.py：One-level instruction (organization) generation code

- main_new_command2_dismantle.py：Two-level instruction task planning code

- main_new_command2_generate.py：Two-level instruction generation code

- main_new_command3_generate.py：Three-level instruction generation and task planning code

- /data：Examples for generating instructions

- /data_to_json_9_8：Code for generating a dataset for instruction fine-tuning

## File configuration

To perform task generation and planning, it is necessary to configure the API for calling the large model in the code files.

```python
url = "https://flag.smarttrot.com/v1/chat/completions"
api_secret_key = ''
```

After task generation or task planning, an xlsx file will be created to store the content. The next step is to use the code in the 'data_to_json_9_8' folder to convert the content of the xlsx file into JSON format for instruction fine-tuning.
