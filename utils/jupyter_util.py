import torch

def extract_sg_embed(objects, relations, text_encoder, tokenizer):
    device = text_encoder.device
    noises = torch.randn([len(objects), 4], device=device)    
    max_relation_per_image = 5
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
            triplets.append(relation_embedding)


    scenegraph_embedding = torch.cat(triplets, dim=0)

    return scenegraph_embedding.unsqueeze(0)