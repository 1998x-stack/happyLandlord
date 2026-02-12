#!/usr/bin/env python
"""
Simple training runner for Happy Landlord 2V2 project

This script provides a simple interface to start the training process
following the user guide recommendations.
"""

import os
import sys
from datetime import datetime

def main():
    print("=" * 60)
    print("Welcome to Happy Landlord 2V2 Training Runner")
    print("=" * 60)
    print(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Import and run the main training
    try:
        from trainer import Trainer
        from utils import init_logging, set_seed
        from config import Config
        from loguru import logger
        
        # Initialize
        init_logging()
        set_seed(42)
        
        print("✅ Configuration loaded successfully")
        print(f"📊 Log directory: {Config.LOG_DIR}")
        print(f"💾 Model directory: {Config.MODEL_DIR}")
        print(f"📈 TensorBoard directory: {Config.TENSORBOARD_DIR}")
        print()
        
        print("🚀 Starting training process...")
        print("-" * 40)
        
        # Create and run trainer
        trainer = Trainer()
        trainer.run_training()
        
        print()
        print("🎉 Training completed successfully!")
        print(f"📁 Check logs in: {Config.LOG_DIR}")
        print(f"💾 Models saved in: {Config.MODEL_DIR}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("\n⚠️  Training interrupted by user")
        print("💾 Training state preserved - you can restart anytime")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def show_usage():
    """Show usage information"""
    print("Usage:")
    print("  python run_training.py")
    print()
    print("Alternative (from main.py):")
    print("  python main.py")
    print()
    print("To monitor training:")
    print("  tensorboard --logdir=logs")
    print()
    print("To view logs:")
    print("  tail -f logs/*/training.log")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        main()