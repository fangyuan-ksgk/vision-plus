import transformers 
from typing import Dict, Sequence, List
import copy, torch, json, os, av
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
from torch.utils.data import Dataset 
import numpy as np

# Question: It's important to note the flow of special token <image>, IMAGE_TOKEN_INDEX, as well as the DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# -- Before the tokenizer is updated with these special tokens (IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN) we should only include place-holder IMAGE_TOKEN_INDEX input the input_ids
# -- Then, we should expand it to tokenizer.encode(DEFAULT_IM_START_TOKEN) + IMAGE_EMBEDDINGs + tokenizer.encode(DEFAULT_IM_END_TOKEN) with updated tokenizer
# -- In this data processing pipeline, we did not update the tokenizer, therefore DEFAULT_IM_START_TOKEN & DEFAULT_IM_END_TOKEN should NOT BE INCLUDED !


def preprocess_llama3(
    conversations,
    frame_counts: List[int],
    tokenizer: transformers.PreTrainedTokenizer,
    default_system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    """ 
    Note that truncation is not done here, it could remove the EOS token, so it's probably more ideal to do it before this function.
    Prepare input_ids & targets for the model after applying chat template. 
    - convert <image> to IMAGE_TOKEN_INDEX and make sure targets are correct
    """
    
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer) # deepcopy to avoid modification of tokenzier (the '<image>' is a placeholder, not included in actual input_ids)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    input_ids, targets = [], []
    
    
    system_message = default_system_message
    for conv in conversations:
        role = conv.get("role") or conv.get("from")
        if role == "system":
            system_message = conv.get("content") or conv.get("value")
            break
        
    
    input_id, target = [], []
    
    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}]) # Begin with system message
    target += [IGNORE_INDEX] * len(input_id) # mask out tokens with IGNORE_INDEX
    
    for conv in conversations:
        try: 
            role = conv["role"]
            content = conv["content"]
        except: 
            role = conv["from"]
            content = conv["value"]
            
        while "<image>" in content: # Handle Video Frame Repetitions
            frame_num = frame_counts.pop(0)
            content = content.replace("<image>", "<image>" * frame_num, 1)
        
        role = roles.get(role, role) # map towards "user" and "assistant"
        
        conv = [{"role" : role, "content" : content}]
        
        if role == "user":
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id 
            target += [IGNORE_INDEX] * len(encode_id)
        elif role == "assistant":
            mask_seq, target_seq = tokenizer.apply_chat_template(conv, tokenize=False).split(content)
            target_seq = content + target_seq
            mask_tokens = tokenizer.encode(mask_seq)[1:] # remove BOS token
            target_tokens = tokenizer.encode(target_seq)
            
            input_id += mask_tokens + target_tokens
            target += [IGNORE_INDEX] * len(mask_tokens) + target_tokens
        else:
            continue # skip over 'system' message
                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        
        input_ids.append(input_id)
        targets.append(target)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )
    
    
def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
                
    # Subsample frames according to desired fps (assuming desired fps is smaller than actual fps)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    fps = total_frame_num / video_time
    sampling_interval = round(fps / data_args.video_fps)
    frame_idx = list(range(0, total_frame_num, sampling_interval))

    if data_args.frames_upbound > 0: # additional subsampling based on interpolation
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

    num_frames_to_sample = len(frame_idx)
    frames = [video_frames[i] for i in frame_idx]
    video = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    frame_time = [frame.time for frame in frames]
    
    return video, video_time, frame_time, num_frames_to_sample

    
class LazySupervisedDataset(Dataset):
    
    def __init__(self, data_args, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args
        self.data = self.load_data(data_args.data_path)

    def load_data(self, data_path):
        """ 
        Assumption here is that under data_path, a json file is provided, with input_ids interleaving text, image and video (with DEFAULT_IMAGE_TOKEN I suppose), 
        although I don't see a 'DEFAULT_VIDEO_TOKEN' here ? Perhaps we should include one ? or is it not used in inference ? It's just a interleaved text & image 
        without text, so a sequence of images, so technically video should be handled as a sequence of DEFAULT_IMAGE_TOKEN ..... who / where is it processed?
        """
        with open(data_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def process_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"]

    def process_video(self, video_path):
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist!")
        
        if os.path.isdir(video_path): # Path is a directory saving frames separately
            frame_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            frame_files.sort()
            
            num_frames_to_sample = self.data_args.frames_upbound if self.data_args.force_sample else 10
            total_frames = len(frame_files)
            sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
            
            video = []
            for idx in sampled_indices:
                frame_path = frame_files[idx]
                try:
                    with Image.open(frame_path) as img:
                        frame = img.convert("RGB")
                        video.append(frame)
                except IOError:
                    print(f"Failed to read frame at path: {frame_path}")
            
            avg_fps = self.data_args.default_fps  # Use a default FPS or get from data_args
            video_time = total_frames / avg_fps
            frame_time = [i/avg_fps for i in sampled_indices]
        else:
            video, video_time, frame_time, num_frames_to_sample = process_video_with_pyav(video_path, self.data_args)
    
        return video, video_time, frame_time, num_frames_to_sample
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """ 
        Interleaved text, image, video
        """
        source = self.data[i]
        conversations = copy.deepcopy(source["conversations"])
        images = []
        frame_counts = []
        if "media" in source:
            for media_file in source["media"]:
                if "image" in media_file:
                    image = self.process_image(os.path.join(self.data_args.image_folder, media_file["image"]))
                    images.append((image, image.size, "image"))
                    frame_counts.append(image.shape[0])
                elif "video" in media_file:
                    video, _, _, _ = self.process_video(os.path.join(self.data_args.video_folder, media_file["video"]))
                    processor = self.image_processor 
                    video_frames = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                    images.append((video_frames, video[0].size, "video"))
                    frame_counts.append(video_frames.shape[0])

        # Process the conversations and create the data dictionary
        data_dict = preprocess_llama3(conversations, frame_counts, self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Add image or video data if present | Unite video & image
        if images:
            data_dict["image"] = images

        data_dict["id"] = source.get("id", i)

        return data_dict