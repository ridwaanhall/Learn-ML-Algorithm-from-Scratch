"""
Comprehensive test runner for all modules in the project.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    test_modules = [
        'test_loss_functions',
        'test_activation_functions', 
        'test_optimization'
    ]
    
    passed = 0
    failed = 0
    
    for module_name in test_modules:
        try:
            print(f"\n📋 Running {module_name}...")
            module = __import__(module_name)
            
            # Run the module's main function
            if hasattr(module, '__main__'):
                exec(open(f'{module_name}.py').read())
            
            print(f"✅ {module_name} completed successfully")
            passed += 1
            
        except Exception as e:
            print(f"❌ {module_name} failed: {str(e)}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
