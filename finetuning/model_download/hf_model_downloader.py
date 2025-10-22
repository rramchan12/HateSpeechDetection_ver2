#!/usr/bin/env python3
"""
HuggingFace Model Downloader

Downloads models from HuggingFace Hub with authentication support.
Supports both public and private models using HF tokens.

Environment Variables:
    HF_TOKEN: HuggingFace API token for accessing private models
    HUGGING_FACE_HUB_TOKEN: Alternative name for HF token
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import snapshot_download, list_models, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


def _get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment variables.
    
    Checks in order:
    1. HF_TOKEN
    2. HUGGING_FACE_HUB_TOKEN
    3. HF_HOME/.huggingface/token
    
    Returns:
        HuggingFace token string or None if not found
    """
    # Check environment variables
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    if token:
        print(f"[OK] Using HuggingFace token from environment variable")
        return token
    
    # Check default token location
    token_file = Path.home() / '.huggingface' / 'token'
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            print(f"[OK] Using HuggingFace token from {token_file}")
            return token
        except Exception as e:
            print(f"[WARNING] Could not read token file: {e}")
    
    return None


def verify_model_access(
    model_name: str,
    token: Optional[str] = None,
    verbose: bool = True
) -> bool:
    """
    Verify that a model exists and is accessible on HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'gpt2', 'meta-llama/Llama-2-7b')
        token: Optional HuggingFace token for private models
        verbose: Whether to print status messages
        
    Returns:
        True if model is accessible, False otherwise
    """
    if token is None:
        token = _get_hf_token()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL ACCESS VERIFICATION")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Token: {'Present' if token else 'Not provided'}")
        print(f"{'='*60}\n")
    
    try:
        # Try to access model card/config
        from huggingface_hub import model_info
        info = model_info(model_name, token=token)
        
        if verbose:
            print(f"[OK] Model found: {model_name}")
            print(f"  Author: {info.author or 'Unknown'}")
            print(f"  Downloads: {info.downloads:,}" if info.downloads else "")
            print(f"  Tags: {', '.join(info.tags[:5])}" if info.tags else "")
            if info.private:
                print(f"  Access: Private (requires authentication)")
            else:
                print(f"  Access: Public")
        
        return True
        
    except RepositoryNotFoundError:
        if verbose:
            print(f"[FAILED] Model not found: {model_name}")
            print(f"  The model does not exist on HuggingFace Hub")
        return False
        
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            if verbose:
                print(f"[FAILED] Authentication required for: {model_name}")
                print(f"  This is a private model or requires a token")
                print(f"  Set HF_TOKEN environment variable or login with: huggingface-cli login")
        else:
            if verbose:
                print(f"[FAILED] HTTP error accessing model: {e}")
        return False
        
    except Exception as e:
        if verbose:
            print(f"[FAILED] Error verifying model access: {e}")
        return False


def download_model(
    model_name: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = True
) -> Optional[str]:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache downloaded models (default: ~/.cache/huggingface)
        token: HuggingFace token for private models (checks env if None)
        force_download: Force re-download even if cached
        resume_download: Resume interrupted downloads
        
    Returns:
        Path to downloaded model directory, or None if download failed
    """
    # Get token from environment if not provided
    if token is None:
        token = _get_hf_token()
    
    # Verify access first
    if not verify_model_access(model_name, token, verbose=True):
        return None
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    if cache_dir:
        print(f"Cache directory: {cache_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Download entire model repository
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
            resume_download=resume_download,
            local_files_only=False
        )
        
        print(f"\n[SUCCESS] Model downloaded successfully!")
        print(f"  Location: {model_path}")
        return model_path
        
    except RepositoryNotFoundError:
        print(f"[FAILED] Model not found: {model_name}")
        return None
        
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"[FAILED] Authentication failed")
            print(f"  Set HF_TOKEN environment variable or login with: huggingface-cli login")
        else:
            print(f"[FAILED] HTTP error: {e}")
        return None
        
    except Exception as e:
        print(f"[FAILED] Error downloading model: {e}")
        return None


def list_available_models(
    search_query: Optional[str] = None,
    limit: int = 20,
    sort: str = "downloads",
    filter_tags: Optional[list] = None
) -> None:
    """
    List available models on HuggingFace Hub.
    
    Args:
        search_query: Search term to filter models
        limit: Maximum number of models to display
        sort: Sort criteria ('downloads', 'likes', 'created')
        filter_tags: List of tags to filter by (e.g., ['text-generation', 'pytorch'])
    """
    print(f"\n{'='*60}")
    print(f"AVAILABLE MODELS ON HUGGINGFACE HUB")
    print(f"{'='*60}")
    if search_query:
        print(f"Search: {search_query}")
    if filter_tags:
        print(f"Tags: {', '.join(filter_tags)}")
    print(f"{'='*60}\n")
    
    try:
        models = list_models(
            search=search_query,
            limit=limit,
            sort=sort,
            filter=filter_tags,
            full=True
        )
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model.id}")
            if model.downloads:
                print(f"   Downloads: {model.downloads:,}")
            if model.tags:
                print(f"   Tags: {', '.join(model.tags[:3])}")
            print()
            
    except Exception as e:
        print(f"[FAILED] Error listing models: {e}")


def cli_main():
    """Command-line interface for the HuggingFace model downloader."""
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub with authentication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify model access
  python hf_model_downloader.py --verify gpt2
  
  # Download a model
  python hf_model_downloader.py --download microsoft/phi-2
  
  # Download with token
  HF_TOKEN=hf_xxxx python hf_model_downloader.py --download meta-llama/Llama-2-7b
  
  # List available models
  python hf_model_downloader.py --list --search "llama" --tags text-generation
  
  # Suggest alternative models
  python hf_model_downloader.py --suggest
        """
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        metavar="MODEL",
        help="Verify access to a model"
    )
    
    parser.add_argument(
        "--download",
        type=str,
        metavar="MODEL",
        help="Download a model from HuggingFace"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Suggest alternative open-source models for hate speech detection"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        help="Search query for listing models"
    )
    
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Filter models by tags"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for downloaded models"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace token (or use HF_TOKEN env variable)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    
    args = parser.parse_args()
    
    # Handle suggestions
    if args.suggest:
        print("\n" + "="*60)
        print("SUGGESTED MODELS FOR HATE SPEECH DETECTION")
        print("="*60)
        print("\nOpen-source models suitable for hate speech classification:\n")
        
        suggestions = [
            ("openai/gpt-oss-20b", "20.9B params, high quality, OSS flagship model"),
            ("openai/gpt-oss-120b", "120B params, best quality, requires more VRAM"),
            ("microsoft/phi-2", "2.7B params, efficient, good for classification"),
            ("microsoft/Phi-3-mini-4k-instruct", "3.8B params, instruction-tuned"),
            ("meta-llama/Llama-3.2-1B", "1B params, fast, requires HF token"),
            ("meta-llama/Llama-3.2-3B", "3B params, good balance, requires HF token"),
            ("mistralai/Mistral-7B-v0.1", "7B params, high quality"),
            ("google/flan-t5-base", "220M params, very fast, good for fine-tuning"),
            ("google/flan-t5-large", "780M params, good performance"),
        ]
        
        for model_id, description in suggestions:
            print(f"  • {model_id}")
            print(f"    {description}")
            print()
        
        print("To verify access: python hf_model_downloader.py --verify MODEL_NAME")
        print("To download: python hf_model_downloader.py --download MODEL_NAME")
        return 0
    
    # Handle verify
    if args.verify:
        success = verify_model_access(args.verify, token=args.token)
        return 0 if success else 1
    
    # Handle download
    if args.download:
        model_path = download_model(
            args.download,
            cache_dir=args.cache_dir,
            token=args.token,
            force_download=args.force
        )
        return 0 if model_path else 1
    
    # Handle list
    if args.list:
        list_available_models(
            search_query=args.search,
            filter_tags=args.tags
        )
        return 0
    
    # If no action specified, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
