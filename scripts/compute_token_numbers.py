import json


# def preprocessing_function_train(examples):
#     max_seq_length = args.max_source_length + args.max_target_length + 1
#     # 添加EOS

#     model_inputs = {
#         "input_ids": [],
#         "labels": [],
#     }

#     # __import__('pdb').set_trace()
#     for i in range(len(examples["input"])):
#         if examples["input"][i] and examples["output"][i] and examples["instruction"]:
#             inputs, outputs, instruction = examples["input"][i], examples["output"][i], examples["instruction"][i]
#             outputs = str(outputs)
#             prompt = instruction + inputs + " ->"

#             a_ids = tokenizer.encode(
#                 text=prompt, add_special_tokens=True, truncation=True, max_length=args.max_source_length
#             )
#             b_ids = tokenizer.encode(
#                 text=outputs, add_special_tokens=False, truncation=True, max_length=args.max_target_length
#             )

#             context_length = len(a_ids)
#             input_ids = a_ids + b_ids + [model.generation_config.eos_token_id]
#             # print(f"===={model.generation_config.pad_token_id}")
#             labels = (
#                 [model.generation_config.pad_token_id] * context_length + b_ids + [model.generation_config.eos_token_id]
#             )
#             # 构建 batch padding
#             pad_len = max_seq_length - len(input_ids)
#             input_ids = input_ids + [model.generation_config.pad_token_id] * pad_len
#             labels = labels + [model.generation_config.pad_token_id] * pad_len
#             if args.ignore_pad_token_for_loss:
#                 labels = [(l if l != model.generation_config.pad_token_id else -100) for l in labels]

#             model_inputs["input_ids"].append(input_ids)
#             model_inputs["labels"].append(labels)
