import torch
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset


def preprocess_scenegraph(examples, text_encoder, tokenizer):
    device = text_encoder.device
    dtype = text_encoder.dtype

    masks_list = []
    positive_embeddings_list = []
    for relationships in examples['relationships']:
        max_relations = 30

        # 1. build list of <subject, predicate, object>
        triplets = []
        for relation in relationships:
            subject_names = relation["subject"]["names"][0]
            predicate = relation["predicate"]
            object_names = relation["object"]["names"][0]


            triplet = [subject_names, predicate, object_names]
            tokenizer_inputs = tokenizer(triplet, padding=True, return_tensors="pt").to(text_encoder.device)

            _relation_embedding = text_encoder(**tokenizer_inputs).pooler_output
            flattened_relation_embedding = _relation_embedding.flatten(start_dim=0).unsqueeze(0)
            triplets.append(flattened_relation_embedding)
        
        if len(triplets) == 0:
            _scenegraph_embedding = torch.zeros(max_relations, 768*3, device=device, dtype=dtype)
        else:
            _scenegraph_embedding = torch.cat(triplets, dim=0)

        n_relations = min(len(_scenegraph_embedding), max_relations)


        scenegraph_embedding = torch.zeros(max_relations, 768*3, device=device, dtype=dtype)
        scenegraph_embedding[:n_relations] = _scenegraph_embedding[:n_relations]

        masks = torch.zeros(max_relations, device=device, dtype=dtype)
        masks[:n_relations] = 1

        masks_list.append(masks)
        positive_embeddings_list.append(scenegraph_embedding)


    examples['masks'] = masks_list
    examples['positive_embeddings'] = positive_embeddings_list


# if __name__ =='__main__':

#     tokenizer = CLIPTokenizer.from_pretrained(
#         '/home/luozhouwang/pretrained_models/StableDiffusion/stable-diffusion-v1-5', subfolder="tokenizer"
#     )

#     text_encoder = CLIPTextModel.from_pretrained(
#         '/home/luozhouwang/pretrained_models/StableDiffusion/stable-diffusion-v1-5', subfolder="text_encoder"
#     )

#     dataset = load_dataset("visual_genome", "relationships_v1.2.0")

#     example = dataset['train'][0]

#     preprocess_scenegraph(example, text_encoder, tokenizer)
