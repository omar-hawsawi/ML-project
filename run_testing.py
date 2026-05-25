from testing_class import SignLanguageTester

# Initialize the tester with the trained TCN model
print("="*60)
print("INITIALIZING TCN MODEL TESTER")
print("="*60)
tester = SignLanguageTester('tcn_best.h5', 'train.csv')

# Test on a batch of samples
print("\n" + "="*60)
print("RUNNING BATCH TEST ON TCN MODEL")
print("="*60)
results = tester.test_batch(base_path='.', num_samples=200)

# Test with confidence filtering
print("\n" + "="*60)
print("RUNNING CONFIDENCE-FILTERED TEST")
print("="*60)
confidence_results = tester.test_with_confidence_filter(base_path='.', confidence_threshold=0.8)

print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
