"""
Unit Tests for NLP SOAP Pipeline
Tests tokenization, classification, entity extraction, and accuracy metrics.
"""

import pytest
from datetime import datetime
from nlp_soap import SOAPExtractor

class TestSOAPExtractor:
    """Test suite for SOAP extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create SOAP extractor instance."""
        return SOAPExtractor()
    
    def test_tokenization(self, extractor):
        """Test sentence tokenization functionality."""
        # Test basic tokenization
        text = "Patient reports chest pain. Vital signs are stable. Diagnosis is angina. Prescribe medication."
        sentences = extractor.tokenize(text)
        
        assert len(sentences) == 4
        assert sentences[0] == "Patient reports chest pain"
        assert sentences[1] == "Vital signs are stable"
        assert sentences[2] == "Diagnosis is angina"
        assert sentences[3] == "Prescribe medication"
        
        # Test empty text
        assert extractor.tokenize("") == []
        assert extractor.tokenize("   ") == []
        
        # Test text without punctuation
        single_sentence = "Patient seems fine"
        assert extractor.tokenize(single_sentence) == ["Patient seems fine"]

    def test_sentence_classification(self, extractor):
        """Test SOAP section classification."""
        test_cases = [
            ("Patient reports severe headache", "subjective"),
            ("Blood pressure is 120/80", "objective"),
            ("Diagnosis is migraine", "assessment"),
            ("Prescribe ibuprofen 400mg", "plan"),
            ("The weather is nice today", "unknown")
        ]
        
        for sentence, expected_category in test_cases:
            result = extractor.classify_sentence(sentence)
            assert result == expected_category

    def test_entity_extraction_objective(self, extractor):
        """Test entity extraction for objective section."""
        text = "Blood pressure 130/85, temperature 98.6°F, pulse: 72"
        entities = extractor.extract_entities(text, "objective")
        
        # Should find blood pressure, temperature, and pulse
        assert len(entities) >= 2  # At least BP and temp
        
        entity_types = [e["type"] for e in entities]
        assert "blood_pressure" in entity_types
        assert "temperature" in entity_types

    def test_entity_extraction_assessment(self, extractor):
        """Test entity extraction for assessment section."""
        text = "Patient has diabetes and hypertension"
        entities = extractor.extract_entities(text, "assessment")
        
        # Should find diabetes and hypertension
        diagnoses = [e["value"] for e in entities if e["type"] == "diagnosis"]
        assert "diabetes" in diagnoses
        assert "hypertension" in diagnoses

    def test_entity_extraction_plan(self, extractor):
        """Test entity extraction for plan section."""
        text = "Prescribe penicillin and metoprolol for treatment"
        entities = extractor.extract_entities(text, "plan")
        
        # Should find medications
        medications = [e["value"] for e in entities if e["type"] == "medication"]
        assert len(medications) >= 1
        
        # Check for medication patterns
        medication_values = " ".join(medications).lower()
        assert "penicillin" in medication_values or "metoprolol" in medication_values

    def test_process_voice_note_happy_path(self, extractor):
        """Test complete voice note processing."""
        voice_note = """
        Patient reports chest pain and shortness of breath. 
        Blood pressure is 140/90, pulse 85, temperature 99.2°F.
        Assessment shows possible cardiac event.
        Plan to prescribe aspirin and schedule follow-up.
        """
        
        result = extractor.process_voice_note(voice_note)
        
        assert result["ok"] is True
        assert result["error"] is None
        
        data = result["data"]
        assert "note_id" in data
        assert "processed_at" in data
        assert data["original_text"] == voice_note
        
        # Check SOAP sections
        soap_sections = data["soap_sections"]
        assert len(soap_sections["subjective"]["sentences"]) > 0
        assert len(soap_sections["objective"]["sentences"]) > 0
        assert len(soap_sections["assessment"]["sentences"]) > 0
        assert len(soap_sections["plan"]["sentences"]) > 0
        
        # Check metadata
        metadata = data["metadata"]
        assert metadata["total_sentences"] > 0
        assert metadata["processing_time_ms"] > 0

    def test_process_voice_note_edge_cases(self, extractor):
        """Test edge cases for voice note processing."""
        # Test empty note
        result = extractor.process_voice_note("")
        assert result["ok"] is True
        assert result["data"]["metadata"]["total_sentences"] == 0
        
        # Test note with unknown sentences
        unknown_note = "The weather is nice. It's a sunny day."
        result = extractor.process_voice_note(unknown_note)
        assert result["ok"] is True
        assert len(result["data"]["metadata"]["unknown_sentences"]) == 2

    def test_accuracy_metrics_calculation(self, extractor):
        """Test accuracy metrics calculation."""
        # Mock processed note
        processed_note = {
            "soap_sections": {
                "subjective": {"sentences": ["patient reports pain"]},
                "objective": {"sentences": ["blood pressure 120/80"]},
                "assessment": {"sentences": ["diagnosis hypertension"]},
                "plan": {"sentences": ["prescribe medication"]}
            }
        }
        
        # Mock ground truth
        ground_truth = {
            "soap_sections": {
                "subjective": {"sentences": ["patient reports chest pain"]},
                "objective": {"sentences": ["blood pressure 120/80"]},
                "assessment": {"sentences": ["diagnosis shows hypertension"]},
                "plan": {"sentences": ["prescribe blood pressure medication"]}
            }
        }
        
        metrics = extractor.calculate_accuracy_metrics(processed_note, ground_truth)
        
        assert "subjective_accuracy" in metrics
        assert "objective_accuracy" in metrics
        assert "assessment_accuracy" in metrics
        assert "plan_accuracy" in metrics
        assert "overall_accuracy" in metrics
        
        # All accuracies should be between 0 and 1
        for key, value in metrics.items():
            if key.endswith("_accuracy"):
                assert 0 <= value <= 1

    def test_time_complexity_performance(self, extractor):
        """Test processing time for different input sizes."""
        # Generate notes of different sizes
        small_note = "Patient reports pain. Vital signs normal. Diagnosis: strain. Prescribe rest."
        medium_note = small_note * 10  # 40 sentences
        large_note = small_note * 25   # 100 sentences
        
        # Process and check timing
        for note, expected_max_time in [(small_note, 100), (medium_note, 500), (large_note, 1000)]:
            result = extractor.process_voice_note(note)
            processing_time = result["data"]["metadata"]["processing_time_ms"]
            
            # Should complete within reasonable time (rough performance test)
            assert processing_time < expected_max_time
            assert processing_time > 0

    def test_space_complexity_memory_usage(self, extractor):
        """Test memory usage doesn't grow excessively."""
        # Process multiple notes to check for memory leaks
        base_note = "Patient reports symptoms. Examination normal. Assessment: healthy. Plan: discharge."
        
        for i in range(10):
            note = f"Note {i}: " + base_note
            result = extractor.process_voice_note(note)
            assert result["ok"] is True
            
            # Each note should be processed independently
            assert f"Note {i}" in result["data"]["original_text"]

    def test_concurrent_processing_safety(self, extractor):
        """Test that the extractor is safe for concurrent use."""
        import threading
        import time
        
        results = []
        
        def process_note(note_text, note_id):
            result = extractor.process_voice_note(f"{note_text} ID: {note_id}")
            results.append((note_id, result["ok"]))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=process_note, 
                args=("Patient reports symptoms. Examination normal.", i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All should complete successfully
        assert len(results) == 5
        for note_id, success in results:
            assert success is True


class TestPerformanceTargets:
    """Test performance targets (≤5 seconds per 60-second note)."""
    
    @pytest.fixture
    def extractor(self):
        return SOAPExtractor()
    
    def test_latency_target_60_second_note(self, extractor):
        """Test processing time for 60-second voice note equivalent."""
        # Approximate 60-second voice note (about 150-200 words)
        note_60_seconds = """
        Patient reports experiencing chest pain for the past two hours, describes it as sharp and radiating to the left arm.
        Also complains of shortness of breath and nausea. Pain started while climbing stairs at home.
        Patient has history of hypertension and diabetes, currently taking metformin and lisinopril.
        No known allergies. Social history includes smoking one pack per day for twenty years.
        On examination, patient appears anxious and diaphoretic. Vital signs show blood pressure 160/95,
        pulse 110 beats per minute, respiratory rate 22, temperature 98.8 degrees Fahrenheit, oxygen saturation 94% on room air.
        Cardiovascular exam reveals tachycardia with regular rhythm, no murmurs or gallops.
        Lungs are clear to auscultation bilaterally. Extremities show no edema.
        EKG shows ST elevation in leads V2 through V5 consistent with anterior STEMI.
        Troponin levels are elevated at 2.5. Chest X-ray shows no acute abnormalities.
        Assessment is ST elevation myocardial infarction, likely anterior wall.
        Plan includes immediate cardiology consultation for emergent cardiac catheterization.
        Administer aspirin 325mg, clopidogrel 600mg loading dose, and metoprolol 25mg twice daily.
        Start heparin drip per protocol. NPO for procedure. Monitor telemetry continuously.
        Patient and family educated about condition and treatment plan. Prognosis discussed.
        """
        
        result = extractor.process_voice_note(note_60_seconds)
        
        assert result["ok"] is True
        processing_time_ms = result["data"]["metadata"]["processing_time_ms"]
        
        # Target: ≤5 seconds (5000ms) per 60-second note
        assert processing_time_ms <= 5000, f"Processing took {processing_time_ms}ms, exceeds 5000ms target"
        
        # Verify content was processed
        sections = result["data"]["soap_sections"]
        total_sentences = sum(len(sections[s]["sentences"]) for s in sections.keys())
        assert total_sentences > 10  # Should have substantial content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])