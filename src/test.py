def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    scheduled_value = slope * t + start_e
    if start_e < end_e:
        return min(max(scheduled_value, start_e), end_e)
    else:
        return max(min(scheduled_value, start_e), end_e)

# Test the scheduler
def test_linear_schedule():
    # Test cases
    tests = [
        {"start_e": 1.0, "end_e": 0.0, "duration": 10, "t_values": [0, 5, 10, 15]},
        {"start_e": 0.0, "end_e": 1.0, "duration": 10, "t_values": [0, 5, 10, 15]},
        {"start_e": 0.5, "end_e": 0.5, "duration": 10, "t_values": [0, 5, 10]},  # Constant value
    ]
    
    # Run tests
    for i, test in enumerate(tests):
        print(f"\nTest {i + 1}: start_e={test['start_e']}, end_e={test['end_e']}, duration={test['duration']}")
        for t in test["t_values"]:
            result = linear_schedule(test["start_e"], test["end_e"], test["duration"], t)
            print(f"  At step {t}: {result:.4f}")

# Run the test
test_linear_schedule()
