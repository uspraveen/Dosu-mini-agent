"""
Simple Smoke Test for Dosu Identity Service
Just tests the basic functions work - no fancy stuff!
"""
import requests
import json
import sys

def test_server_running():
    """Test 1: Is the server running?"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running at http://localhost:8000")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Server is not running!")
        print("   Start it with: uvicorn main:app --reload")
        return False

def test_identity_resolution():
    """Test 2: Basic identity resolution"""
    try:
        payload = {
            "provider": "github",
            "external_id": "test_user_123",
            "email": "test@example.com"
        }

        response = requests.post("http://localhost:8000/resolve",
                               json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if "user_id" in data:
                print("✅ Identity resolution works")
                print(f"   Created user: {data['user_id']}")
                return True
            else:
                print("❌ Identity resolution returned bad data")
                print(f"   Response: {data}")
                return False
        else:
            print(f"❌ Identity resolution failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Identity resolution error: {e}")
        return False

def test_ticket_creation():
    """Test 3: Basic ticket creation"""
    try:
        # First get a user_id
        user_payload = {
            "provider": "slack",
            "external_id": "test_user_456",
            "email": "ticket_test@example.com"
        }
        user_response = requests.post("http://localhost:8000/resolve",
                                    json=user_payload, timeout=10)

        if user_response.status_code != 200:
            print("❌ Couldn't create user for ticket test")
            return False

        user_id = user_response.json()["user_id"]

        # Now create a ticket
        ticket_payload = {
            "user_id": user_id,
            "source": "github_issue",
            "external_id": "test_issue_123",
            "title": "Test issue for smoke test",
            "body": "This is a test issue to verify ticket creation works"
        }

        response = requests.post("http://localhost:8000/ticket",
                               json=ticket_payload, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if "ticket_id" in data:
                print("✅ Ticket creation works")
                print(f"   Created ticket: {data['ticket_id']}")
                return True
            else:
                print("❌ Ticket creation returned bad data")
                print(f"   Response: {data}")
                return False
        else:
            print(f"❌ Ticket creation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Ticket creation error: {e}")
        return False

def main():
    print("🚀 Simple Dosu Smoke Test")
    print("=" * 40)

    tests = [
        test_server_running,
        test_identity_resolution,
        test_ticket_creation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line between tests
        except KeyboardInterrupt:
            print("\n⏹️  Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()

    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your system works!")
    else:
        print("💥 Some tests failed. Check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)