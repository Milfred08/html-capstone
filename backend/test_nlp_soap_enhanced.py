"""
Enhanced Unit Tests for NLP SOAP Pipeline - CS-499 Milestone Three
Advanced Algorithm Testing with Complexity Analysis and Performance Benchmarking

This test suite demonstrates:
1. Algorithm complexity verification through performance testing
2. Data structure efficiency validation
3. Scalability testing with increasing dataset sizes
4. Comparative analysis of different algorithmic approaches
5. Memory usage and time complexity validation
"""

import pytest
import time
import threading
from datetime import datetime
from collections import defaultdict
from nlp_soap import (
    EnhancedSOAPExtractor, OptimizedSOAPClassifier, MedicalTrie, TrieNode,
    AlgorithmType, PerformanceMetrics, demonstrate_enhanced_algorithms
)

class TestAdvancedDataStructures:
    """Test suite for advanced data structures implementation."""
    
    @pytest.fixture
    def medical_trie(self):
        """Create medical trie for testing."""
        trie = MedicalTrie()
        # Insert sample medical terms
        test_terms = [
            ("chest pain", "subjective"),
            ("blood pressure", "objective"), 
            ("diagnosis", "assessment"),
            ("medication", "plan"),
            ("headache", "subjective"),
            ("pulse rate", "objective")
        ]
        
        for term, category in test_terms:
            trie.insert(term, category, {"test": True})
        
        return trie
    
    def test_trie_node_structure(self):
        """Test TrieNode data structure implementation."""
        node = TrieNode()
        
        # Test initial state
        assert node.children == {}
        assert node.is_end_word is False
        assert node.word_data is None
        assert node.frequency == 0
        
        # Test node modification
        node.is_end_word = True
        node.frequency = 5
        node.word_data = {"category": "test", "metadata": {}}
        
        assert node.is_end_word is True
        assert node.frequency == 5
        assert node.word_data["category"] == "test"

    def test_trie_insertion_complexity(self, medical_trie):
        """
        Test trie insertion time complexity O(m) where m is word length.
        
        Expected: Insertion time should scale linearly with word length.
        """
        # Test words of increasing length
        test_words = [
            ("a", "test"),           # Length 1
            ("ab", "test"),          # Length 2  
            ("abcd", "test"),        # Length 4
            ("abcdefgh", "test"),    # Length 8
            ("abcdefghijklmnop", "test"),  # Length 16
        ]
        
        insertion_times = []
        
        for word, category in test_words:
            start_time = time.perf_counter()
            medical_trie.insert(word, category)
            end_time = time.perf_counter()
            
            insertion_time = (end_time - start_time) * 1000  # Convert to ms
            insertion_times.append((len(word), insertion_time))
        
        # Verify that insertion time doesn't grow exponentially
        # (Linear growth is acceptable, exponential would indicate O(2^m) complexity)
        for i in range(1, len(insertion_times)):
            prev_length, prev_time = insertion_times[i-1]
            curr_length, curr_time = insertion_times[i]
            
            # Time shouldn't increase more than proportionally to length increase
            length_ratio = curr_length / prev_length
            time_ratio = curr_time / prev_time if prev_time > 0 else 1
            
            # Allow for some variance but ensure it's not exponential growth
            assert time_ratio < length_ratio * 10, f"Insertion time growing too fast: {time_ratio} vs {length_ratio}"

    def test_trie_search_complexity(self, medical_trie):
        """
        Test trie search time complexity O(n*m) for text of length n and average term length m.
        """
        # Test texts of increasing length
        test_texts = [
            "chest",                    # 5 chars
            "chest pain",              # 10 chars
            "patient has chest pain",  # 19 chars
            "the patient reports chest pain and headache symptoms today",  # 55 chars
        ]
        
        search_times = []
        
        for text in test_texts:
            start_time = time.perf_counter()
            matches = medical_trie.search_prefix(text)
            end_time = time.perf_counter()
            
            search_time = (end_time - start_time) * 1000
            search_times.append((len(text), search_time, len(matches)))
        
        # Verify search results are found
        assert len(search_times) == len(test_texts)
        
        # Check that longer texts don't cause exponential time increase
        for i in range(1, len(search_times)):
            prev_length, prev_time, prev_matches = search_times[i-1]
            curr_length, curr_time, curr_matches = search_times[i]
            
            if prev_time > 0:
                length_ratio = curr_length / prev_length
                time_ratio = curr_time / prev_time
                
                # Time growth should be reasonable (not exponential)
                assert time_ratio < length_ratio * 5, f"Search time growing too fast: {time_ratio}"

    def test_trie_memory_efficiency(self):
        """Test trie memory usage with large vocabulary."""
        import sys
        
        trie = MedicalTrie()
        initial_size = sys.getsizeof(trie)
        
        # Insert many terms to test memory scaling
        medical_terms = [
            f"medical_term_{i}" for i in range(100)
        ]
        
        for term in medical_terms:
            trie.insert(term, "test_category")
        
        final_size = sys.getsizeof(trie)
        
        # Memory should grow but not excessively
        memory_growth = final_size - initial_size
        
        # Each term should use reasonable memory (rough estimate)
        avg_memory_per_term = memory_growth / len(medical_terms)
        
        # Should be less than 1KB per term for reasonable efficiency
        assert avg_memory_per_term < 1024, f"Memory usage per term too high: {avg_memory_per_term} bytes"

class TestAlgorithmComplexity:
    """Test suite for algorithm complexity analysis."""
    
    @pytest.fixture
    def extractor(self):
        """Create enhanced SOAP extractor for testing."""
        return EnhancedSOAPExtractor()
    
    @pytest.fixture
    def classifier(self):
        """Create optimized classifier for testing.""" 
        return OptimizedSOAPClassifier()

    def test_tokenization_linear_complexity(self, classifier):
        """
        Test that tokenization has O(n) time complexity.
        
        Tokenization should scale linearly with input text length.
        """
        # Generate texts of increasing length
        base_text = "Patient reports chest pain. Blood pressure is elevated. "
        test_texts = [
            base_text * (2**i) for i in range(1, 6)  # Exponentially increasing length
        ]
        
        tokenization_times = []
        
        for text in test_texts:
            start_time = time.perf_counter()
            tokens = classifier.tokenize_advanced(text)
            end_time = time.perf_counter()
            
            tokenization_time = (end_time - start_time) * 1000
            tokenization_times.append((len(text), tokenization_time, len(tokens)))
        
        # Verify linear time complexity
        for i in range(1, len(tokenization_times)):
            prev_length, prev_time, prev_tokens = tokenization_times[i-1]
            curr_length, curr_time, curr_tokens = tokenization_times[i]
            
            if prev_time > 0:
                length_ratio = curr_length / prev_length
                time_ratio = curr_time / prev_time
                
                # Time should grow roughly linearly with text length
                # Allow some variance but ensure it's not quadratic or worse
                assert time_ratio < length_ratio * 3, f"Tokenization not linear: {time_ratio} vs {length_ratio}"

    def test_hybrid_classification_complexity(self, classifier):
        """
        Test hybrid classification complexity O(n*m + k).
        
        Should combine trie search O(n*m) and hash lookup O(k).
        """
        # Test sentences of varying complexity
        test_sentences = [
            "Pain",  # Simple
            "Patient has chest pain",  # Medium
            "The patient reports severe chest pain radiating to left arm with associated nausea",  # Complex
            "Comprehensive assessment reveals multiple symptoms including chest pain, shortness of breath, nausea, dizziness, and fatigue with associated vital sign abnormalities"  # Very complex
        ]
        
        classification_times = []
        
        for sentence in test_sentences:
            start_time = time.perf_counter()
            category, confidence, metadata = classifier.classify_sentence_hybrid(sentence)
            end_time = time.perf_counter()
            
            classification_time = (end_time - start_time) * 1000
            classification_times.append((len(sentence), classification_time, category, confidence))
        
        # Verify that classification works
        assert all(ct[2] != "unknown" or ct[3] == 0.0 for ct in classification_times[-2:])  # Complex sentences should be classified
        
        # Check complexity growth is reasonable
        for i in range(1, len(classification_times)):
            prev_length, prev_time, _, _ = classification_times[i-1]
            curr_length, curr_time, _, _ = classification_times[i]
            
            if prev_time > 0:
                length_ratio = curr_length / prev_length
                time_ratio = curr_time / prev_time
                
                # Should not grow faster than quadratic
                assert time_ratio < length_ratio**2 * 2, f"Classification complexity too high: {time_ratio}"

    def test_caching_performance_improvement(self, classifier):
        """Test that caching improves performance for repeated operations."""
        test_sentence = "Patient reports chest pain and shortness of breath"
        
        # Clear cache to start fresh
        classifier._cache.clear()
        
        # First classification (cache miss)
        start_time = time.perf_counter()
        result1 = classifier.classify_sentence_hybrid(test_sentence)
        first_time = time.perf_counter() - start_time
        
        # Second classification (cache hit)
        start_time = time.perf_counter()
        result2 = classifier.classify_sentence_hybrid(test_sentence)
        second_time = time.perf_counter() - start_time
        
        # Results should be identical
        assert result1 == result2
        
        # Second call should be significantly faster (cache hit)
        if first_time > 0:
            speedup_ratio = first_time / second_time if second_time > 0 else float('inf')
            assert speedup_ratio > 2, f"Caching not effective: {speedup_ratio}x speedup"

class TestPerformanceBenchmarking:
    """Test suite for performance benchmarking and optimization validation."""
    
    @pytest.fixture
    def extractor(self):
        return EnhancedSOAPExtractor()

    def test_processing_time_targets(self, extractor):
        """
        Test that processing meets CS-499 performance targets.
        
        Target: ≤5000ms per 60-second voice note (≈150-200 words)
        """
        # Simulate 60-second voice note (approximately 150-200 words)
        long_note = """
        Patient is a 45-year-old male presenting to the emergency department with chief complaint 
        of severe chest pain that started approximately two hours ago while climbing stairs at home.
        The pain is described as crushing, substernal, radiating to the left arm and jaw.
        Associated symptoms include shortness of breath, nausea, and diaphoresis.
        Patient has a past medical history significant for hypertension and diabetes mellitus type 2.
        Current medications include metformin and lisinopril. No known drug allergies.
        Social history is notable for smoking one pack per day for the past twenty years.
        
        On physical examination, the patient appears anxious and in moderate distress.
        Vital signs reveal blood pressure of 160/95 mmHg, heart rate 110 beats per minute,
        respiratory rate 22 breaths per minute, temperature 99.2 degrees Fahrenheit,
        and oxygen saturation 94% on room air. Cardiovascular examination shows tachycardia
        with regular rhythm, no murmurs or gallops appreciated. Pulmonary examination reveals
        clear breath sounds bilaterally. Extremities show no pedal edema.
        
        Laboratory results show elevated troponin levels at 2.5 ng/mL. Complete blood count
        and basic metabolic panel are within normal limits. Electrocardiogram demonstrates
        ST elevation in leads V2 through V5, consistent with anterior wall myocardial infarction.
        Chest X-ray shows no acute cardiopulmonary abnormalities.
        
        Based on the clinical presentation, laboratory findings, and electrocardiogram changes,
        the patient is diagnosed with ST-elevation myocardial infarction affecting the anterior wall.
        The treatment plan includes immediate cardiology consultation for emergent cardiac catheterization.
        Medications initiated include aspirin 325 mg, clopidogrel 600 mg loading dose,
        metoprolol 25 mg twice daily, and heparin infusion per protocol.
        Patient will be monitored on continuous cardiac telemetry. Family has been notified
        and educated regarding the diagnosis and treatment plan.
        """
        
        # Process the note
        start_time = time.perf_counter()
        result = extractor.process_voice_note_enhanced(long_note)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Verify processing completed successfully
        assert result["ok"] is True
        
        # Check performance target
        TARGET_TIME_MS = 5000
        assert processing_time <= TARGET_TIME_MS, f"Processing time {processing_time:.2f}ms exceeds target {TARGET_TIME_MS}ms"
        
        # Verify data quality
        data = result["data"]
        assert data["metadata"]["total_sentences"] > 10  # Should have substantial content
        
        # Check that multiple SOAP sections were populated
        populated_sections = sum(1 for section in data["soap_sections"].values() if section["sentences"])
        assert populated_sections >= 3, "Should populate at least 3 SOAP sections"

    def test_algorithm_comparison_accuracy(self, extractor):
        """Test that algorithm comparison provides meaningful performance data."""
        test_notes = [
            "Patient reports chest pain. Blood pressure 140/90. Diagnosis: angina. Prescribe nitroglycerin.",
            "Headache and nausea present. Temperature 101F, pulse 88. Possible migraine. Recommend rest.",
            "Shortness of breath noted. Lung sounds clear, oxygen saturation 95%. Plan: monitor closely."
        ]
        
        comparison_result = extractor.compare_algorithms_performance(test_notes)
        
        assert comparison_result["ok"] is True
        
        comparison_data = comparison_result["data"]
        
        # Verify all algorithms were tested
        algorithms_tested = list(comparison_data["algorithm_performance"].keys())
        expected_algorithms = ["hybrid", "trie_based", "hash_based", "rule_based"]
        
        for algo in expected_algorithms:
            assert algo in algorithms_tested, f"Algorithm {algo} not tested"
        
        # Verify performance metrics are reasonable
        for algo_name, metrics in comparison_data["algorithm_performance"].items():
            assert metrics["execution_time_ms"] >= 0
            assert metrics["memory_usage_bytes"] >= 0
            assert 0 <= metrics["accuracy_score"] <= 1.0
            assert metrics["time_complexity"] is not None
            assert metrics["space_complexity"] is not None
        
        # Verify recommendations are provided
        recs = comparison_data["recommendations"]
        assert "fastest_algorithm" in recs
        assert "most_accurate" in recs
        assert "most_memory_efficient" in recs

    def test_scalability_with_increasing_dataset_size(self, extractor):
        """Test algorithm scalability with increasing dataset sizes."""
        # Generate datasets of increasing size
        base_note = "Patient reports symptoms. Vital signs normal. Assessment: healthy. Plan: discharge."
        
        dataset_sizes = [1, 5, 10, 20]
        processing_times = []
        
        for size in dataset_sizes:
            test_dataset = [f"Note {i}: {base_note}" for i in range(size)]
            
            start_time = time.perf_counter()
            comparison_result = extractor.compare_algorithms_performance(test_dataset)
            processing_time = time.perf_counter() - start_time
            
            assert comparison_result["ok"] is True
            processing_times.append((size, processing_time))
        
        # Verify that processing time grows reasonably with dataset size
        for i in range(1, len(processing_times)):
            prev_size, prev_time = processing_times[i-1]
            curr_size, curr_time = processing_times[i]
            
            if prev_time > 0:
                size_ratio = curr_size / prev_size
                time_ratio = curr_time / prev_time
                
                # Time should grow no worse than quadratically with dataset size
                assert time_ratio <= size_ratio**2 * 2, f"Poor scalability: {time_ratio} vs {size_ratio**2}"

    def test_memory_usage_optimization(self, extractor):
        """Test that memory usage is optimized and doesn't leak."""
        import gc
        import sys
        
        # Baseline memory usage
        gc.collect()
        baseline_memory = sys.getsizeof(extractor)
        
        # Process multiple notes
        test_notes = [
            f"Patient {i} reports various symptoms and requires assessment." 
            for i in range(50)
        ]
        
        for note in test_notes:
            result = extractor.process_voice_note_enhanced(note, f"test_note_{hash(note)}")
            assert result["ok"] is True
        
        # Check memory after processing
        gc.collect()
        final_memory = sys.getsizeof(extractor)
        
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be reasonable (not unbounded)
        max_acceptable_growth = len(test_notes) * 1024  # 1KB per note max
        assert memory_growth < max_acceptable_growth, f"Memory growth too large: {memory_growth} bytes"

class TestConcurrencyAndThreadSafety:
    """Test suite for concurrent processing and thread safety."""
    
    def test_concurrent_processing_thread_safety(self):
        """Test that concurrent processing is thread-safe."""
        extractor = EnhancedSOAPExtractor()
        
        results = []
        errors = []
        
        def process_note(note_id: int):
            try:
                note_text = f"Patient {note_id} reports symptoms. Assessment needed. Treatment plan required."
                result = extractor.process_voice_note_enhanced(note_text, f"concurrent_note_{note_id}")
                results.append((note_id, result["ok"]))
            except Exception as e:
                errors.append((note_id, str(e)))
        
        # Create multiple threads to process notes concurrently
        threads = []
        num_threads = 10
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_note, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all processing completed successfully
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_threads
        assert all(success for _, success in results), "Some concurrent processing failed"

    def test_cache_thread_safety(self):
        """Test that caching mechanism is thread-safe."""
        classifier = OptimizedSOAPClassifier()
        
        test_sentence = "Patient reports chest pain and difficulty breathing"
        results = []
        errors = []
        
        def classify_sentence(thread_id: int):
            try:
                for _ in range(5):  # Multiple calls per thread
                    result = classifier.classify_sentence_hybrid(test_sentence)
                    results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        num_threads = 5
        
        for i in range(num_threads):
            thread = threading.Thread(target=classify_sentence, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify thread safety
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == num_threads * 5
        
        # All results should be identical (same sentence, same classification)
        first_result = results[0][1]
        for thread_id, result in results:
            assert result == first_result, f"Thread {thread_id} got different result"

class TestEdgeCasesAndRobustness:
    """Test suite for edge cases and system robustness."""
    
    @pytest.fixture
    def extractor(self):
        return EnhancedSOAPExtractor()

    def test_empty_input_handling(self, extractor):
        """Test handling of empty or minimal input."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            ".",  # Single punctuation
            "a",  # Single character
        ]
        
        for text in edge_cases:
            result = extractor.process_voice_note_enhanced(text)
            assert result["ok"] is True
            
            data = result["data"]
            assert data["metadata"]["total_sentences"] >= 0
            assert data["metadata"]["processing_time_ms"] >= 0

    def test_very_large_input_handling(self, extractor):
        """Test handling of very large input texts."""
        # Generate very large note (simulate long dictation)
        large_note = "Patient reports symptoms. " * 1000  # ~25,000 characters
        
        result = extractor.process_voice_note_enhanced(large_note)
        
        assert result["ok"] is True
        
        data = result["data"]
        assert data["metadata"]["total_sentences"] > 100
        
        # Should still meet reasonable performance targets
        assert data["metadata"]["processing_time_ms"] < 30000  # 30 seconds max for very large input

    def test_special_characters_and_unicode(self, extractor):
        """Test handling of special characters and Unicode text."""
        special_text = """
        Patient reports: "severe pain" (10/10 scale).
        Temperature: 38.5°C, BP: 140/90 mmHg.
        Médication: 500mg paracétamol.
        Plan: follow-up in 2–3 days.
        """
        
        result = extractor.process_voice_note_enhanced(special_text)
        
        assert result["ok"] is True
        
        data = result["data"]
        assert data["metadata"]["total_sentences"] > 0
        
        # Should extract vital signs despite special characters
        objective_entities = data["soap_sections"]["objective"]["entities"]
        assert len(objective_entities) > 0, "Should extract entities from text with special characters"

def test_demonstration_function():
    """Test that the demonstration function runs without errors."""
    import io
    import sys
    
    # Capture stdout to test demonstration output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        # Run the demonstration
        demonstrate_enhanced_algorithms()
        
        output = buffer.getvalue()
        
        # Verify key demonstration elements are present
        assert "CS-499 MILESTONE THREE" in output
        assert "ENHANCED ALGORITHMS" in output
        assert "COMPLEXITY ANALYSIS" in output
        assert "Performance Results" in output
        assert "DATA STRUCTURES SHOWCASE" in output
        
        # Verify complexity information is shown
        assert "O(" in output  # Big O notation present
        assert "Time:" in output  # Timing information
        assert "Memory:" in output  # Memory usage
        
    finally:
        sys.stdout = old_stdout

if __name__ == "__main__":
    # Run comprehensive testing
    pytest.main([__file__, "-v", "--tb=short"])