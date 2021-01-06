import aspectlib
import aspectlib.debug
import inspect


@aspectlib.Aspect
def mock_print_log_to_console(*args, **kwargs):
    if kwargs["message_type"] == 'before':
        yield aspectlib.Return('Function {} from class {} was called.'.format(kwargs['function'],
                                                                              kwargs['function_class']))
    elif kwargs["message_type"] == 'after':
        yield aspectlib.Return('Getting {} for {}. \n'.format(kwargs['function'].split("from_")[1].replace("_", " "),
                                                              kwargs['function'].split("_")[1]))


@aspectlib.Aspect
def transform_data_from_dataframe_to_dict_for_texts(*args, **kwargs):
    stack = inspect.stack()
    called_function_class = None
    for index in range(len(list(stack[2][0].f_locals)) - 1, -1, -1):
        if "<class '" in str(stack[2][0].f_locals.get(list(stack[2][0].f_locals)[index])):
            called_function_class = str(stack[2][0].f_locals.get(list(stack[2][0].f_locals)[index]))
            break
    called_function = str(stack[2].code_context[0].split(".")[stack[2].code_context[0].count(".")].split("(", )[0])

    df_data = yield aspectlib.Proceed

    with aspectlib.weave(print_logs_to_console, mock_print_log_to_console):
        print(print_logs_to_console(message_type='before', function=called_function,
                                    function_class=called_function_class))

        dict_data = {"id": [], "label": [], "text": []}
        for index_line, data_line in df_data.iterrows():
            dict_data.get("id").append(int(data_line["id"]))
            dict_data.get("text").append(data_line["text"])
            if 'test' not in str(called_function):
                dict_data.get("label").append(data_line["label"])

        print(print_logs_to_console(message_type='after', function=called_function,
                                    function_class=called_function_class))

    yield aspectlib.Return(dict_data)


@aspectlib.Aspect
def transform_data_from_dataframe_to_list_for_images(*args, **kwargs):
    stack = inspect.stack()
    called_function_class = None
    for index in range(len(list(stack[2][0].f_locals)) - 1, -1, -1):
        if "<class '" in str(stack[2][0].f_locals.get(list(stack[2][0].f_locals)[index])):
            called_function_class = str(stack[2][0].f_locals.get(list(stack[2][0].f_locals)[index]))
            break
    called_function = str(stack[2].code_context[0].split(".")[stack[2].code_context[0].count(".")].split("(", )[0])

    df_data = yield aspectlib.Proceed

    with aspectlib.weave(print_logs_to_console, mock_print_log_to_console):
        print(print_logs_to_console(message_type='before', function=called_function,
                                    function_class=called_function_class))

        list_data = []
        for index_line, data_line in df_data.iterrows():
            list_data.append("Implementation/data/{}".format(str(data_line["img"])))

        print(print_logs_to_console(message_type='after', function=called_function,
                                    function_class=called_function_class))

    yield aspectlib.Return(list_data)


def print_logs_to_console():
    pass
