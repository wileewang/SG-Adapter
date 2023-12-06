import torch
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
import json
import os
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

def visualize_attention(attention_map, end_token_idx, end_relation_idx):
    """
    Visualizes the attention map using a heatmap.

    Args:
    - attention_map (torch.Tensor): The attention map of shape [num_heads, sequence_length, num_relations].
    - end_token_idx (int): The index until which tokens are to be visualized.
    - end_relation_idx (int): The index until which relations are to be visualized.
    """

    # Summarize along the attention heads
    summarized_attention = attention_map.sum(dim=0, keepdim=True)

    # Extract the desired portion based on the specified indices
    desired_attention = summarized_attention[0, :end_token_idx, :end_relation_idx].numpy()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(desired_attention, cmap="YlGnBu", cbar=True, annot=True, fmt=".2f")
    plt.xlabel('Relation Index')
    plt.ylabel('Token Index')
    plt.title('Attention Heatmap')
    plt.show()

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # print(line)  # Add this line to print each line before parsing
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, filename): 
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def draw_scene_graph(objs, triples, **kwargs):
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]

    for i, obj in enumerate(objs):
        if ignore_dummies and obj == 'image':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        s_idx = int(s)
        o_idx = int(o)
        if ignore_dummies and p == 'in' and objs[o_idx] == 'image':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s_idx, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o_idx, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    
    return None

def extract_sg_embed(objects, relations, text_encoder, tokenizer, wo_triplet=True):
    device = text_encoder.device
    noises = torch.randn([len(objects), 4], device=device)    
    max_relation_per_image = 10
    triplets = []
    for i in range(max_relation_per_image):
        if i <= len(relations) - 1:
            relation = relations[i]
            # 1. build list of <subject, predicate, object>

            subject_names = objects[int(relation[0])]
            predicate = relation[1]
            object_names = objects[int(relation[2])]


            triplet = [subject_names, predicate, object_names]
            tokenizer_inputs = tokenizer(triplet, padding=True, return_tensors="pt").to(device)
            tokens_embedding = text_encoder(**tokenizer_inputs).pooler_output


            normalized_sub_category_embed = torch.nn.functional.normalize(tokens_embedding[0], p=2, dim=0)
            sub_embed = torch.cat([normalized_sub_category_embed, noises[int(relation[0])]]).view(1, -1)


            pre_embed = tokens_embedding[1].view(1, -1)

            normalized_obj_category_embed = torch.nn.functional.normalize(tokens_embedding[2], p=2, dim=0)
            obj_embed = torch.cat([normalized_obj_category_embed, noises[int(relation[2])]]).view(1, -1)

            relation_embedding = torch.cat([sub_embed, pre_embed, obj_embed], dim=1)
            if wo_triplet:
                triplets.append(pre_embed)
            else:
                triplets.append(relation_embedding)
        else:
            triplet = ["", "", ""]
            tokenizer_inputs = tokenizer(triplet, padding=True, return_tensors="pt").to(device)
            tokens_embedding = text_encoder(**tokenizer_inputs).pooler_output


            normalized_sub_category_embed = torch.nn.functional.normalize(tokens_embedding[0], p=2, dim=0)
            sub_embed = torch.cat([normalized_sub_category_embed, torch.zeros([4], device=device)]).view(1, -1)


            pre_embed = tokens_embedding[1].view(1, -1)

            normalized_obj_category_embed = torch.nn.functional.normalize(tokens_embedding[2], p=2, dim=0)
            obj_embed = torch.cat([normalized_obj_category_embed, torch.zeros([4], device=device)]).view(1, -1)

            relation_embedding = torch.cat([sub_embed, pre_embed, obj_embed], dim=1)
            if wo_triplet:
                triplets.append(pre_embed)
            else:
                triplets.append(relation_embedding)


    scenegraph_embedding = torch.cat(triplets, dim=0)

    return scenegraph_embedding.unsqueeze(0)

def generate_sg_attention_mask(mapping, batch_size=1, text_length=77, num_relations=10, dtype=torch.float16):
    # the token idx start from 0 which we missed the start_token, so we need to add 1 to token_idx

    mask_value = torch.finfo(dtype).min
    # Initial mask filled with -1e9 (i.e., prevent attention everywhere)
    mask = torch.full((batch_size, text_length, num_relations), mask_value, dtype=dtype)

    # For each mapping in the form {relation_idx: [token_idx1, token_idx2, ...]}
    for relation_idx, token_idx_list in enumerate(mapping):
        if relation_idx >= 10:
            break
        if len(token_idx_list) == 0:
            continue
        for token_idx in token_idx_list:
            # Set the mask value for the mapped relation to 1 for that token (i.e., allow attention)
            mask[:, token_idx, int(relation_idx)] = 0
    return mask

def generate_clip_attention_mask(mappings, batch_size=1, text_length=77, dtype=torch.float16):
    """
    Make causal mask used for bi-directional self-attention.
    
    Args:
    - mappings (list of list of int): List of token groupings for grouped attention.
    - batch_size (int): Batch size for the mask tensor.
    - text_length (int): Maximum sequence length.
    - dtype (torch.dtype): Data type of the mask.
    
    Returns:
    - torch.Tensor: Causal attention mask.
    """
    mask_value = torch.finfo(dtype).min
    mask = torch.full((text_length, text_length), mask_value, dtype=dtype)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    for mapping in mappings:
        # Make sure each group only attends to itself
        for i in mapping:
            for j in range(text_length):
                if j not in mapping or j > i:
                    mask[i, j] = mask_value

    # Find the maximum number/token in the mappings list
    if any(mappings):  # Check if there's at least one non-empty sublist
        last_token_in_mapping = max(max(sublist) for sublist in mappings if sublist)  # sublist is non-empty
    else:
        last_token_in_mapping = 0  # Default value if all sublists are empty

    # Any token after the final token in the mapping should attend to all previous tokens
    for i in range(last_token_in_mapping+1, text_length):
        for j in range(i):
            mask[i, j] = 0
    
    mask[:, 0] = 0

    return mask[None, None, :, :].expand(batch_size, 1, text_length, text_length)

def generate_encoder_attention_mask(input_mask, sequence_length=77):
    """
    Generate an encoder attention mask with padding.

    Args:
    - input_mask (`torch.Tensor`): Input tensor mask with ones and zeros.
    - sequence_length (int): Length to which the sequence should be padded.

    Returns:
    - encoder_attention_mask (`torch.Tensor`): The padded mask tensor of shape `(batch_size, sequence_length)`.
    """
    # Convert ones to True and zeros to False
    bool_mask = input_mask.bool()
    
    # Pad the mask to the desired sequence length
    padding_size = sequence_length - input_mask.size(1)
    padding = torch.full((input_mask.size(0), padding_size), False, dtype=torch.bool)
    encoder_attention_mask = torch.cat([bool_mask, padding], dim=1)
    
    return encoder_attention_mask

def generate_self_attention_mask(mapping, batch_size=1, text_length=77, dtype=torch.float16):
    # Initialize the mask with False values
    self_attention_mask = torch.full((batch_size, text_length, text_length), False, dtype=torch.bool)
    
    # Iterate over the groups and set the corresponding positions to True
    for group in mapping:
        for i in group:
            for j in group:
                self_attention_mask[:, i, j] = True

    # Now, convert the boolean mask to the appropriate float values
    # Convert True to 0.0 (can attend) and False to -inf (cannot attend)
    self_attention_mask_float = torch.full(self_attention_mask.size(), torch.finfo(dtype).min, dtype=dtype)
    self_attention_mask_float[self_attention_mask] = 0.0

    return self_attention_mask_float

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    scenegraph_embeddings = torch.stack([example["scenegraph_embedding"] for example in examples])
    output = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "scenegraph_embeddings": scenegraph_embeddings,
    }
    if "sg_attention_mask" in examples[0].keys():
        sg_attention_masks = torch.stack([example["sg_attention_mask"] for example in examples])
        output['sg_attention_masks'] = sg_attention_masks
    if "self_attention_mask" in examples[0].keys():
        self_attention_masks = torch.stack([example["self_attention_mask"] for example in examples])
        output['self_attention_masks'] = self_attention_masks
    if "clip_attention_mask" in examples[0].keys():
        clip_attention_masks = torch.stack([example["clip_attention_mask"] for example in examples])
        output['clip_attention_masks'] = clip_attention_masks
    if "encoder_attention_mask" in examples[0].keys():
        encoder_attention_masks = torch.stack([example["encoder_attention_mask"] for example in examples])
        output['encoder_attention_masks'] = encoder_attention_masks

    return output

def preprocess_scenegraph(examples, text_encoder, tokenizer, args):
    device = text_encoder.device
    dtype = text_encoder.dtype

    scenegraph_embedding_list = []
    shuffle_indices_list = []
    for objects, relations in zip(examples['objects'], examples['relations']):
        noises = torch.randn([len(objects), 4], device=device)    
        max_relation_per_image = 10
        triplets = []
        for i in range(max_relation_per_image):
            if i <= len(relations) - 1:
                relation = relations[i]
                # 1. build list of <subject, predicate, object>

                subject_names = objects[int(relation[0])]
                predicate = relation[1]
                object_names = objects[int(relation[2])]


                triplet = [subject_names, predicate, object_names]
                tokenizer_inputs = tokenizer(triplet, padding=True, return_tensors="pt").to(device)
                tokens_embedding = text_encoder(**tokenizer_inputs).pooler_output


                normalized_sub_category_embed = torch.nn.functional.normalize(tokens_embedding[0], p=2, dim=0)
                sub_embed = torch.cat([normalized_sub_category_embed, noises[int(relation[0])]]).view(1, -1)


                pre_embed = tokens_embedding[1].view(1, -1)

                normalized_obj_category_embed = torch.nn.functional.normalize(tokens_embedding[2], p=2, dim=0)
                obj_embed = torch.cat([normalized_obj_category_embed, noises[int(relation[2])]]).view(1, -1)

                relation_embedding = torch.cat([sub_embed, pre_embed, obj_embed], dim=1)
                if args.wo_linear:
                    triplets.append(pre_embed)
                else:
                    triplets.append(relation_embedding)
            else:
                triplet = ["", "", ""]
                tokenizer_inputs = tokenizer(triplet, padding=True, return_tensors="pt").to(device)
                tokens_embedding = text_encoder(**tokenizer_inputs).pooler_output


                normalized_sub_category_embed = torch.nn.functional.normalize(tokens_embedding[0], p=2, dim=0)
                sub_embed = torch.cat([normalized_sub_category_embed, torch.zeros([4], device=device)]).view(1, -1)


                pre_embed = tokens_embedding[1].view(1, -1)

                normalized_obj_category_embed = torch.nn.functional.normalize(tokens_embedding[2], p=2, dim=0)
                obj_embed = torch.cat([normalized_obj_category_embed, torch.zeros([4], device=device)]).view(1, -1)

                relation_embedding = torch.cat([sub_embed, pre_embed, obj_embed], dim=1)
                if args.wo_linear:
                    triplets.append(pre_embed)
                else:
                    triplets.append(relation_embedding)

        if args.shuffle:
            indices = list(range(len(triplets)))
            random.shuffle(indices)
            shuffle_indices_list.append(indices)
            triplets = [triplets[i] for i in indices]
        scenegraph_embedding = torch.cat(triplets, dim=0)
        scenegraph_embedding_list.append(scenegraph_embedding)
    examples['scenegraph_embedding'] = scenegraph_embedding_list

    if args.use_encoder_attn_mask:
        encoder_attention_mask_list = []
        for caption in examples['caption']: 
            encoder_attention_mask = generate_encoder_attention_mask(
                tokenizer(
                    caption, padding=True, return_tensors="pt"
                    ).attention_mask
                ).squeeze(0).to(device)
            encoder_attention_mask_list.append(encoder_attention_mask)
        examples['encoder_attention_mask'] = encoder_attention_mask_list

    
    if args.use_sg_attn_mask:
        sg_attention_mask_list = []
        for idx, mapping in enumerate(examples['mapping']):
            while len(mapping) < max_relation_per_image:
                mapping.append([])
            if args.shuffle:
                indices = shuffle_indices_list[idx]
                mapping = [mapping[i] for i in indices]
            sg_attention_mask = generate_sg_attention_mask(mapping, batch_size=1, text_length=77, num_relations=max_relation_per_image, dtype=dtype).squeeze(0).to(device)
            sg_attention_mask_list.append(sg_attention_mask)

        examples['sg_attention_mask'] = sg_attention_mask_list
    
    if args.use_self_attn_mask:
        self_attention_mask_list = []
        for idx, mapping in enumerate(examples['mapping']):
            while len(mapping) < max_relation_per_image:
                mapping.append([])
            if args.shuffle:
                indices = shuffle_indices_list[idx]
                mapping = [mapping[i] for i in indices]
            self_attention_mask = generate_self_attention_mask(mapping, batch_size=1, text_length=77, dtype=dtype).squeeze(0).to(device)
            self_attention_mask_list.append(self_attention_mask)

        examples['self_attention_mask'] = self_attention_mask_list

    if args.use_clip_attn_mask:
        clip_attention_mask_list = []
        for idx, mapping in enumerate(examples['mapping']):
            while len(mapping) < max_relation_per_image:
                mapping.append([])
            if args.shuffle:
                indices = shuffle_indices_list[idx]
                mapping = [mapping[i] for i in indices]
            clip_attention_mask = generate_clip_attention_mask(mapping, batch_size=1, text_length=77, dtype=dtype).squeeze(0).to(device)
            clip_attention_mask_list.append(clip_attention_mask)
        examples['clip_attention_mask'] = clip_attention_mask_list

