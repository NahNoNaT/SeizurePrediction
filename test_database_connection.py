#!/usr/bin/env python3
"""
Script to test DATABASE_URL connection and diagnose issues
"""
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import psycopg

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


def test_database_url():
    """Test the DATABASE_URL connection"""
    
    database_url = os.getenv("DATABASE_URL", "").strip()
    
    print("=" * 70)
    print("DATABASE CONNECTION TEST")
    print("=" * 70)
    
    # Check if DATABASE_URL is set
    if not database_url:
        print("❌ ERROR: DATABASE_URL is not set")
        print("   Add DATABASE_URL to your .env file or environment variables")
        return False
    
    print(f"✓ DATABASE_URL is set")
    
    # Parse the URL
    try:
        parsed = urlparse(database_url)
        print(f"\n📋 Connection Details:")
        print(f"   - Host: {parsed.hostname}")
        print(f"   - Port: {parsed.port or 5432}")
        print(f"   - Database: {parsed.path.lstrip('/')}")
        print(f"   - User: {parsed.username}")
        print(f"   - SSL Mode: {parsed.query.split('sslmode=')[1] if 'sslmode=' in parsed.query else 'not specified'}")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse DATABASE_URL: {e}")
        return False
    
    # Test connection
    print(f"\n🔄 Testing connection to database...")
    try:
        # Try to connect
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"✓ Connected successfully!")
                print(f"   PostgreSQL version: {version[0]}")
        
        # Test again with a simple query
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                if result[0] == 1:
                    print(f"✓ Database query test passed")
        
        return True
        
    except psycopg.OperationalError as e:
        print(f"❌ Connection failed (Operational Error):")
        error_msg = str(e)
        print(f"   {error_msg}")
        
        # Provide specific troubleshooting
        if "host" in error_msg.lower() or "connection refused" in error_msg.lower():
            print(f"\n💡 Possible solutions:")
            print(f"   - Check if Supabase server is running")
            print(f"   - Verify the hostname in DATABASE_URL")
            print(f"   - Check your internet connection")
            print(f"   - Verify Supabase firewall settings (if applicable)")
        elif "password" in error_msg.lower() or "authentication" in error_msg.lower():
            print(f"\n💡 Possible solutions:")
            print(f"   - Check if password is correct")
            print(f"   - Make sure password doesn't contain special characters that need escaping")
            print(f"   - Verify credentials in Supabase dashboard")
        elif "database" in error_msg.lower():
            print(f"\n💡 Possible solutions:")
            print(f"   - Verify database name is correct")
            print(f"   - Check if database exists in Supabase")
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}")
        print(f"   {str(e)}")
        return False


def test_environment():
    """Check environment and dependencies"""
    print("\n" + "=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check psycopg
    try:
        import psycopg
        print(f"✓ psycopg is installed (version: {psycopg.__version__})")
    except ImportError:
        print(f"❌ psycopg is not installed")
        print(f"   Run: pip install psycopg")
        return False
    
    # Check DATABASE_URL
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        print(f"✓ DATABASE_URL environment variable is set")
    else:
        print(f"❌ DATABASE_URL environment variable is not set")
    
    return True


if __name__ == "__main__":
    # Load .env file first
    env_file_loaded = load_env_file()
    if env_file_loaded:
        print("✓ Loaded .env file\n")
    else:
        print("⚠️  Warning: Could not find .env file\n")
    
    env_ok = test_environment()
    print()
    
    if env_ok:
        success = test_database_url()
        
        print("\n" + "=" * 70)
        if success:
            print("✅ All checks passed! Your DATABASE_URL is working correctly.")
        else:
            print("❌ Database connection test failed. Check the errors above.")
        print("=" * 70)
        
        sys.exit(0 if success else 1)
    else:
        print("\n" + "=" * 70)
        print("❌ Environment check failed. Please fix the issues above.")
        print("=" * 70)
        sys.exit(1)
