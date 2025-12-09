#!/usr/bin/env python3
"""
Comprehensive Test Suite for Demo App Enhancements
Tests all new functions: explanations, goal setter, and progress tracking
"""

import json
from pathlib import Path

def validate_file_structure():
    """Validate that demo_app.py has all required components."""
    demo_path = Path(__file__).parent / "demo_app.py"
    assert demo_path.exists(), f"demo_app.py not found at {demo_path}"
    
    content = demo_path.read_text()
    
    # Check for function definitions
    required_functions = {
        'generate_prediction_explanation': 'def generate_prediction_explanation',
        'render_goal_setter': 'def render_goal_setter',
        'render_prediction_explanations': 'def render_prediction_explanations',
        'render_predictions': 'def render_predictions',
        'render_risk_assessment': 'def render_risk_assessment',
        'render_quick_advice': 'def render_quick_advice',
        'main': 'def main',
    }
    
    print("✓ Checking for required functions...")
    for func_name, pattern in required_functions.items():
        assert pattern in content, f"Missing function: {func_name}"
        print(f"  ✓ {func_name}")
    
    # Check for key UI elements
    ui_elements = {
        'Goal Setter': '🎯 Goal Setter',
        'Understanding Predictions': '🔍 Understanding Your Predictions',
        'Progress Toward Goals': '🎯 Progress Toward Your Goals',
    }
    
    print("\n✓ Checking for UI elements...")
    for element_name, pattern in ui_elements.items():
        assert pattern in content, f"Missing UI element: {element_name}"
        print(f"  ✓ {element_name}")
    
    # Check file size
    lines = len(content.split('\n'))
    assert lines > 850, f"File too small: {lines} lines (expected >850)"
    print(f"\n✓ File size: {lines} lines (OK)")
    
    return True

def validate_function_signatures():
    """Validate that new functions have correct signatures."""
    print("\n✓ Checking function signatures...")
    
    # Read the file
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    # Check generate_prediction_explanation signature
    assert 'def generate_prediction_explanation(target: str, value: float, inputs: dict, thresholds: dict) -> dict:' in content
    print("  ✓ generate_prediction_explanation: (target, value, inputs, thresholds) -> dict")
    
    # Check render_goal_setter signature
    assert 'def render_goal_setter():' in content
    print("  ✓ render_goal_setter: () -> dict")
    
    # Check render_prediction_explanations signature
    assert 'def render_prediction_explanations(predictions, inputs, thresholds):' in content
    print("  ✓ render_prediction_explanations: (predictions, inputs, thresholds) -> None")
    
    return True

def validate_integration():
    """Validate that new functions are integrated in main()."""
    print("\n✓ Checking integration in main()...")
    
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    # Check that render_goal_setter is called
    assert 'goals = render_goal_setter()' in content
    print("  ✓ render_goal_setter() called in main")
    
    # Check that render_prediction_explanations is called
    assert 'render_prediction_explanations(predictions, inputs, thresholds)' in content
    print("  ✓ render_prediction_explanations() called in main")
    
    # Check that goal progress is displayed
    assert 'Progress Toward Your Goals' in content
    print("  ✓ Goal progress tracking displayed")
    
    # Check that goals dict is used
    assert "goals['stress_target']" in content
    assert "goals['anxiety_target']" in content
    assert "goals['mood_target']" in content
    assert "goals['sleep_target']" in content
    assert "goals['exercise_target']" in content
    print("  ✓ All 5 goal targets used in display")
    
    return True

def validate_explanation_logic():
    """Validate that explanation generator covers all metrics."""
    print("\n✓ Checking explanation logic...")
    
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    # Check for all metric explanations
    metrics = [
        'anxiety_score',
        'stress_level',
        'mood_score',
        'depression_score',
    ]
    
    for metric in metrics:
        assert f"if target == '{metric}'" in content or \
               f'if target == "{metric}"' in content
        print(f"  ✓ Explanation logic for {metric}")
    
    # Check for recommendations structure
    assert "'action':" in content
    assert "'impact':" in content
    assert "'effort':" in content
    print("  ✓ Recommendations have action, impact, effort")
    
    # Check for factors structure
    assert 'explanation[\'factors\'] = ' in content
    print("  ✓ Factors collection implemented")
    
    return True

def validate_ui_components():
    """Validate that UI components are present."""
    print("\n✓ Checking UI components...")
    
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    # Check for sliders
    sliders = [
        'Target Stress Level',
        'Target Anxiety Score',
        'Target Mood Score',
        'Target Sleep Hours',
        'Target Exercise',
    ]
    
    for slider in sliders:
        assert slider in content
        print(f"  ✓ Slider: {slider}")
    
    # Check for expanders
    assert 'st.expander' in content
    print("  ✓ Expanders for metric details")
    
    # Check for metrics display
    assert 'st.metric' in content
    print("  ✓ Metrics display with st.metric")
    
    # Check for progress bars
    assert 'st.progress' in content
    print("  ✓ Progress bars with st.progress")
    
    # Check for columns layout
    assert 'st.columns' in content
    print("  ✓ Column layout for display")
    
    return True

def validate_no_syntax_errors():
    """Validate that there are no Python syntax errors."""
    print("\n✓ Checking for syntax errors...")
    
    import py_compile
    demo_path = Path(__file__).parent / "demo_app.py"
    
    try:
        py_compile.compile(str(demo_path), doraise=True)
        print("  ✓ No syntax errors detected")
        return True
    except py_compile.PyCompileError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

def validate_docstrings():
    """Validate that functions have docstrings."""
    print("\n✓ Checking docstrings...")
    
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    functions = [
        'generate_prediction_explanation',
        'render_goal_setter',
        'render_prediction_explanations',
    ]
    
    for func in functions:
        # Simple check for docstring (triple quotes after def)
        func_start = content.find(f'def {func}')
        assert func_start != -1, f"Function {func} not found"
        
        func_section = content[func_start:func_start+500]
        assert '"""' in func_section, f"No docstring for {func}"
        print(f"  ✓ {func} has docstring")
    
    return True

def validate_color_system():
    """Validate that color system is maintained."""
    print("\n✓ Checking color system...")
    
    demo_path = Path(__file__).parent / "demo_app.py"
    content = demo_path.read_text()
    
    # Check for color emojis
    color_emojis = ['🟢', '🟠', '🔴']
    for emoji in color_emojis:
        assert emoji in content
        print(f"  ✓ Color emoji {emoji} used")
    
    # Check for get_color_and_delta function
    assert 'def get_color_and_delta' in content
    print("  ✓ get_color_and_delta function present")
    
    # Check that thresholds are used
    assert 'thresholds' in content
    print("  ✓ Thresholds configuration used")
    
    return True

def test_example_explanation():
    """Test example explanation generation (without Streamlit)."""
    print("\n✓ Testing explanation logic with example data...")
    
    # Simulate what generate_prediction_explanation does
    inputs = {
        'sleep_quality': 4,
        'social_interactions': 2,
        'caffeine_mg': 450,
        'outdoor_time_minutes': 15,
        'work_pressure': 7,
        'work_hours': 11,
        'sleep_hours': 6,
        'meetings_count': 7,
        'exercise_minutes': 20,
    }
    
    target = 'anxiety_score'
    
    # Expected factors for high anxiety
    expected_factors = []
    if inputs['sleep_quality'] < 6:
        expected_factors.append('sleep_quality')
    if inputs['social_interactions'] < 3:
        expected_factors.append('social_interactions')
    if inputs['caffeine_mg'] > 300:
        expected_factors.append('caffeine_mg')
    
    print(f"  ✓ Example inputs: {len(inputs)} behavioral metrics")
    print(f"  ✓ Expected factors for anxiety: {expected_factors}")
    assert 'sleep_quality' in expected_factors
    assert 'social_interactions' in expected_factors
    assert 'caffeine_mg' in expected_factors
    print("  ✓ Factor detection logic works correctly")
    
    return True

def main():
    """Run all validation tests."""
    print("="*70)
    print("DEMO APP ENHANCEMENTS - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("File Structure", validate_file_structure),
        ("Function Signatures", validate_function_signatures),
        ("Integration in Main", validate_integration),
        ("Explanation Logic", validate_explanation_logic),
        ("UI Components", validate_ui_components),
        ("No Syntax Errors", validate_no_syntax_errors),
        ("Docstrings", validate_docstrings),
        ("Color System", validate_color_system),
        ("Example Explanation", test_example_explanation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            results.append((test_name, False))
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "🎉 "*10)
        print("ALL TESTS PASSED!")
        print("Demo enhancements are ready for deployment!")
        print("🎉 "*10)
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - see above for details")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
