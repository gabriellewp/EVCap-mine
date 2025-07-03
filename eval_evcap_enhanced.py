"""
Enhanced EVCap evaluation script that uses flow matching for feature enhancement.
This demonstrates the complete pipeline from training to inference to captioning.
"""

import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms

from models.evcap import EVCap
from train_evcap_flowmatching import ConditionalFlow
from enhance_features import enhance_features, load_models
from search import beam_search


def load_enhanced_evcap_pipeline(evcap_model_path, flow_model_path, device='cuda'):
    """Load the complete enhanced captioning pipeline."""
    
    # Load EVCap model
    model_type = "lmsys/vicuna-13b-v1.5"
    evcap_model = EVCap(
        ext_path='ext_data/ext_memory_lvis.pkl',
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        num_query_token_txt=8,
        topn=9,
        llama_model=model_type,
        prompt_path="prompts/prompt_evcap.txt",
        prompt_template='###Human: {} ###Assistant: ',
        max_txt_len=128,
        end_sym='\n',
        low_resource=False,
        device_8bit=0,
    )
    
    # Load EVCap checkpoint
    evcap_checkpoint = torch.load(evcap_model_path, map_location=device)
    evcap_model.load_state_dict(evcap_checkpoint['model'], strict=False)
    evcap_model = evcap_model.to(device)
    evcap_model.eval()
    
    # Load flow model for feature enhancement
    flow_model = ConditionalFlow(dim=768, cond_dim=768)
    flow_checkpoint = torch.load(flow_model_path, map_location=device)
    if 'flow_model' in flow_checkpoint:
        flow_model.load_state_dict(flow_checkpoint['flow_model'])
    else:
        flow_model.load_state_dict(flow_checkpoint)
    flow_model = flow_model.to(device)
    flow_model.eval()
    
    return evcap_model, flow_model


class EnhancedEVCap:
    """Wrapper that combines EVCap with flow-based feature enhancement."""
    
    def __init__(self, evcap_model, flow_model, device='cuda'):
        self.evcap_model = evcap_model
        self.flow_model = flow_model
        self.device = device
        
        # Feature projection (same as used in training)
        # EVCap's encode_img returns features projected to llama_model.config.hidden_size (5120)
        # We need to project these to 768 for the flow model
        self.feature_projection = torch.nn.Linear(5120, 768).to(device).half()
        
        # Inverse projection to convert enhanced features back to original dimension
        self.inverse_projection = torch.nn.Linear(768, 5120).to(device).half()
    
    def generate_caption(self, image_tensor, use_enhancement=True, num_enhancement_steps=50):
        """
        Generate caption with optional feature enhancement.
        
        Args:
            image_tensor: Preprocessed image tensor
            use_enhancement: Whether to use flow-based feature enhancement
            num_enhancement_steps: Number of enhancement steps for flow model
        """
        with torch.no_grad():
            # Step 1: Extract original Q-former features
            original_features, attn_masks = self.evcap_model.encode_img(image_tensor)
            
            if use_enhancement:
                # Step 2: Project features to flow model dimension
                projected_features = self.feature_projection(original_features)
                
                # Step 3: Apply learned feature enhancement
                enhanced_features = enhance_features(
                    self.flow_model, 
                    projected_features, 
                    attn_masks, 
                    num_steps=num_enhancement_steps,
                    device=self.device
                )
                
                # Step 4: Use enhanced features for captioning
                caption = self._generate_caption_with_features(image_tensor, enhanced_features, attn_masks)
            else:
                # Standard captioning without enhancement (follow eval_evcap.py pattern)
                qform_all_proj, atts_qform_all_proj = self.evcap_model.encode_img(image_tensor)
                prompt_embeds, atts_prompt = self.evcap_model.prompt_wrap(
                    qform_all_proj, atts_qform_all_proj, self.evcap_model.prompt_list
                )
                
                self.evcap_model.llama_tokenizer.padding_side = "right"
                batch_size = qform_all_proj.shape[0]
                bos = torch.ones([batch_size, 1], device=self.device) * self.evcap_model.llama_tokenizer.bos_token_id
                bos = bos.long()
                bos_embeds = self.evcap_model.llama_model.model.embed_tokens(bos)
                embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
                
                caption = beam_search(
                    embeddings=embeddings, 
                    tokenizer=self.evcap_model.llama_tokenizer, 
                    beam_width=5, 
                    model=self.evcap_model.llama_model
                )
                caption = caption[0] if isinstance(caption, list) else caption
            
            return caption
    
    def _generate_caption_with_features(self, image_tensor, enhanced_features, atts_qform_all_proj):
        """
        Generate caption using pre-computed enhanced features.
        This follows the same pattern as the eval_evcap.py script.
        """
        # Project enhanced features back to original dimension (768 -> 5120)
        enhanced_features_proj = self.inverse_projection(enhanced_features)
        
        # Create attention mask for enhanced features
        # atts_enhanced = torch.ones(enhanced_features_proj.size()[:-1], dtype=torch.long).to(self.device)
        
        # Wrap features with prompt (same as original EVCap pipeline)
        prompt_embeds, atts_prompt = self.evcap_model.prompt_wrap(
            enhanced_features_proj, atts_qform_all_proj, self.evcap_model.prompt_list
        )
        
        # Prepare tokenizer and generate BOS token
        self.evcap_model.llama_tokenizer.padding_side = "right"
        batch_size = enhanced_features_proj.shape[0]
        bos = torch.ones([batch_size, 1], device=self.device) * self.evcap_model.llama_tokenizer.bos_token_id
        bos = bos.long()
        bos_embeds = self.evcap_model.llama_model.model.embed_tokens(bos)
        
        # Concatenate embeddings
        embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
        
        # Use beam search for caption generation
        sentence = beam_search(
            embeddings=embeddings, 
            tokenizer=self.evcap_model.llama_tokenizer, 
            beam_width=5, 
            model=self.evcap_model.llama_model
        )
        return sentence[0] if isinstance(sentence, list) else sentence


def compare_captions(image_path, evcap_model_path, flow_model_path, device='cuda'):
    """Compare captions with and without feature enhancement."""
    #image_path is the image path to be captioned
    #evcap_model_path is the path to the EVCap model checkpoint
    #flow_model_path is the path to the flow model checkpoint
    
    # Load models
    evcap_model, flow_model = load_enhanced_evcap_pipeline(evcap_model_path, flow_model_path, device)
    enhanced_captioner = EnhancedEVCap(evcap_model, flow_model, device)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate captions
    print("Generating captions...")
    original_caption = enhanced_captioner.generate_caption(image_tensor, use_enhancement=False)
    enhanced_caption = enhanced_captioner.generate_caption(image_tensor, use_enhancement=True)
    
    print(f"\nOriginal Caption: {original_caption}")
    print(f"Enhanced Caption: {enhanced_caption}")
    
    return original_caption, enhanced_caption


####TODO: add validation for whoops, nocaps, and flickr30k like the eval_evcap.py script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to image for captioning")
    parser.add_argument("--evcap_model", type=str, required=True, help="Path to EVCap model checkpoint")
    parser.add_argument("--flow_model", type=str, required=True, help="Path to flow model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Compare captions with and without enhancement
    compare_captions(args.image_path, args.evcap_model, args.flow_model, args.device)
