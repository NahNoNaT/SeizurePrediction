#!/usr/bin/env python3
"""
Script to test app initialization similar to how FastAPI app starts
"""
import os
import sys
from pathlib import Path

# Load .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
        return True
    return False

if __name__ == "__main__":
    load_env_file()
    
    print("=" * 70)
    print("APP INITIALIZATION TEST")
    print("=" * 70)
    
    try:
        print("\n1️⃣  Importing FastAPI and dependencies...")
        from fastapi import FastAPI
        print("   ✓ FastAPI imported")
        
        print("\n2️⃣  Importing app modules...")
        from app.config import runtime_config
        print("   ✓ Config loaded")
        
        print("\n3️⃣  Testing PostgreSQL store import...")
        try:
            from app.services.postgres_store import PostgresClinicalCaseStore
            print("   ✓ PostgresClinicalCaseStore imported")
            
            print("\n4️⃣  Initializing database store...")
            database_url = os.getenv("DATABASE_URL", "").strip()
            if database_url:
                store = PostgresClinicalCaseStore(database_url)
                print("   ✓ PostgresClinicalCaseStore instance created")
                
                print("\n5️⃣  Testing store initialization...")
                store.initialize()
                print("   ✓ Store initialized successfully")
                print("   ✓ All database tables created/verified")
            else:
                print("   ❌ DATABASE_URL not set")
                sys.exit(1)
        
        except ImportError as e:
            print(f"   ⚠️  PostgresClinicalCaseStore not available: {e}")
            print("   (This is OK if psycopg is optional in your setup)")
        
        except Exception as e:
            print(f"   ❌ Error: {type(e).__name__}: {e}")
            sys.exit(1)
        
        print("\n6️⃣  Testing inference service...")
        from app.services.inference import SeizureInferenceService
        print("   ✓ SeizureInferenceService imported")
        
        print("\n7️⃣  Testing clinical workflow service...")
        from app.services.clinical_workflow import ClinicalAnalysisService
        print("   ✓ ClinicalAnalysisService imported")
        
        print("\n" + "=" * 70)
        print("✅ All initialization tests passed!")
        print("=" * 70)
        print("\n💡 Your app should start successfully.")
        print("\n⚠️  If Render still shows errors:")
        print("   - Check Render's deployment logs for detailed error messages")
        print("   - Verify environment variables are set correctly in Render dashboard")
        print("   - Make sure Supabase allows connections from Render's IP")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR during initialization:")
        print(f"   {type(e).__name__}: {e}")
        print("\n" + "=" * 70)
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
