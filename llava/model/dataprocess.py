import transformers 
from typing import Dict, Sequence
import copy, torch, json, os, av
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
from torch.utils.data import Dataset 
import numpy as np

# Question: It's important to note the flow of special token <image>, IMAGE_TOKEN_INDEX, as well as the DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# -- Before the tokenizer is updated with these special tokens (IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN) we should only include place-holder IMAGE_TOKEN_INDEX input the input_ids
# -- Then, we should expand it to tokenizer.encode(DEFAULT_IM_START_TOKEN) + IMAGE_EMBEDDINGs + tokenizer.encode(DEFAULT_IM_END_TOKEN) with updated tokenizer
# -- In this data processing pipeline, we did not update the tokenizer, therefore DEFAULT_IM_START_TOKEN & DEFAULT_IM_END_TOKEN should NOT BE INCLUDED !

# def preprocess_multimodal(sources: Sequence[str]) -> Dict:
#     # Preprocess input sequence
#     # One IMAGE_TOKEN corresponds to one image
#     for source in sources:
#         for sentence in source:
#             replace_token = DEFAULT_IMAGE_TOKEN
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
#     return sources


def preprocess_llama3(
    sources,
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
    
    for source in sources:
        
        # Extract system message
        system_message = None
        for conv in source:
            role = conv.get("role") or conv.get("from")
            if role == "system":
                system_message = conv.get("content") or conv.get("value")
                break
                        
        if system_message is None:
            system_message = default_system_message
                
        if source[0]["from"] != "human":
            source = source[1:]

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}]) # Begin with system message
        target += [IGNORE_INDEX] * len(input_id) # mask out tokens with IGNORE_INDEX

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role) # map towards "user" and "assistant"
            
            conv = [{"role" : role, "content" : content}]
            
            if role == "user":
                encode_id = tokenizer.apply_chat_template(conv)[1:]
                input_id += encode_id 
                target += [IGNORE_INDEX] * len(encode_id)
            elif role == "assistant":
                mask_seq, target_seq = tokenizer.apply_chat_template(conv, tokenize=False).split(content)
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
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    
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
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

    def process_video(self, video_path, num_frames=10):
        container = av.open(video_path)
        video = container.streams.video[0]
        total_frames = video.frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_image()
                frames.append(self.image_processor(images=img, return_tensors="pt")["pixel_values"][0])
            if len(frames) == num_frames:
                break

        container.close()
        return torch.stack(frames)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """ 
        Lazy means process data when it's loaded | extra latency when first loaded, but faster for subsequent access
        """
        source = self.data[i]
        
        if "image" in source:
            image_file = source["image"]
            image_folder = self.data_args.image_folder
            processor = self.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            conversations = copy.deepcopy([source["conversations"]])
       
        elif "video" in source:
            video_file = source["video"]
            video_folder = self.data_args.video_folder
            video_path = os.path.join(video_folder, video_file)
            if not os.path.exists(video_path):
                print(f"File {video_path} does not exist!")
                return self._get_item(i + 1)

            try:
                # Check if it's a directory of frames or a video file
                if os.path.isdir(video_path):
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

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                
                # Use multiple DEFAULT_IMAGE_TOKENs for video
                video_tokens = DEFAULT_IMAGE_TOKEN * num_frames_to_sample
                
                frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
                
                if self.data_args.add_time_instruction:
                    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time_str}. Please answer the following questions related to this video."
                    source["conversations"][0]["value"] = f'{video_tokens}\n{time_instruction}\n{source["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                else:
                    source["conversations"][0]["value"] = f'{video_tokens}\n{source["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'

                image = [(image, video[0].size, "video")]
                conversations = copy.deepcopy([source["conversations"]])
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_path}")
                return self._get_item(i + 1)
        else:
            conversations = copy.deepcopy([source["conversations"]])

        # Process the conversations and create the data dictionary
        data_dict = preprocess_llama3(conversations, self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # Add image or video data if present
        if "image" in source or "video" in source:
            data_dict["image"] = image

        data_dict["id"] = source.get("id", i)

        return data_dict