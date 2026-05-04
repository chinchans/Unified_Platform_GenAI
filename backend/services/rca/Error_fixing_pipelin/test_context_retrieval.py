#!/usr/bin/env python3
"""
Test script for context-aware retrieval functionality
"""

import json
from .error_handling_pipeline import ErrorHandlingPipeline

def test_context_aware_retrieval():
    """Test the context-aware retrieval with deployment context."""
    
    print("🔍 Testing Context-Aware Retrieval")
    print("=" * 50)
    
    # Initialize pipeline
    print("📦 Initializing pipeline...")
    pipeline = ErrorHandlingPipeline()
    
    # Load deployment context
    print("📄 Loading deployment context...")
    with open('deployment_context.json', 'r') as f:
        deployment_context = json.load(f)
    
    print(f"✅ Deployment context loaded:")
    print(f"   Role: {deployment_context.get('role', 'Unknown')}")
    print(f"   Active configs: {len(deployment_context.get('active_configs', []))}")
    print(f"   Log anchors: {len(deployment_context.get('log_anchors', []))}")
    
    # Test error
    error_text = "No AMF associated to gNB"
    print(f"\n🔍 Testing with error: {error_text}")
    
    # Retrieve context-aware candidates
    print("\n🚀 Running context-aware retrieval...")
    results = pipeline.retrieve_candidates_with_context(
        error_text=error_text,
        deployment_context=deployment_context,
        top_k=10
    )
    
    # Display results
    print("\n📊 RESULTS:")
    print("=" * 30)
    
    print(f"\n🔧 Functions found: {len(results['functions'])}")
    for i, func in enumerate(results['functions'][:5], 1):
        print(f"  {i}. {func['name']} (score: {func['score']:.3f})")
        print(f"     File: {func['file_path']}")
        if func['score'] != func['original_score']:
            print(f"     Boosted from: {func['original_score']:.3f}")
    
    print(f"\n⚙️  Configs found: {len(results['configs'])}")
    for i, config in enumerate(results['configs'][:5], 1):
        print(f"  {i}. {config['param']} = {config['value']} (score: {config['score']:.3f})")
        print(f"     File: {config['file_path']}")
        if config['score'] != config['original_score']:
            print(f"     Boosted from: {config['original_score']:.3f}")
    
    # Save results
    output_file = "output/context_aware_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n✅ Context-aware retrieval test completed!")

if __name__ == "__main__":
    test_context_aware_retrieval()
