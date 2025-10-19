import { useEffect, useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Enhanced Repository Demo Components
const AnimalShelterDemo = () => {
  const [animals, setAnimals] = useState([]);
  const [newAnimal, setNewAnimal] = useState({ name: '', animal_type: 'dog', breed: '', age: 1 });
  const [loading, setLoading] = useState(false);

  const fetchAnimals = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/animals`);
      if (response.data.ok) {
        setAnimals(response.data.data);
      }
    } catch (e) {
      console.error('Error fetching animals:', e);
    } finally {
      setLoading(false);
    }
  };

  const createAnimal = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      const response = await axios.post(`${API}/animals`, newAnimal);
      if (response.data.ok) {
        setNewAnimal({ name: '', animal_type: 'dog', breed: '', age: 1 });
        fetchAnimals(); // Refresh list
      }
    } catch (e) {
      console.error('Error creating animal:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnimals();
  }, []);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Animal Shelter Repository Demo</h2>
      
      {/* Create Animal Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Add New Animal</h3>
        <form onSubmit={createAnimal} className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <input
            type="text"
            placeholder="Animal Name"
            value={newAnimal.name}
            onChange={(e) => setNewAnimal({...newAnimal, name: e.target.value})}
            className="border rounded px-3 py-2"
            required
            data-testid="animal-name-input"
          />
          <select
            value={newAnimal.animal_type}
            onChange={(e) => setNewAnimal({...newAnimal, animal_type: e.target.value})}
            className="border rounded px-3 py-2"
            data-testid="animal-type-select"
          >
            <option value="dog">Dog</option>
            <option value="cat">Cat</option>
            <option value="bird">Bird</option>
            <option value="rabbit">Rabbit</option>
          </select>
          <input
            type="text"
            placeholder="Breed"
            value={newAnimal.breed}
            onChange={(e) => setNewAnimal({...newAnimal, breed: e.target.value})}
            className="border rounded px-3 py-2"
            data-testid="animal-breed-input"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
            data-testid="add-animal-btn"
          >
            {loading ? 'Adding...' : 'Add Animal'}
          </button>
        </form>
      </div>

      {/* Animals List */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-semibold">Animals in Shelter</h3>
          <p className="text-sm text-gray-600">Enhanced with validation, audit trails, and safe operations</p>
        </div>
        <div className="p-6">
          {loading && animals.length === 0 ? (
            <div className="text-center py-4" data-testid="loading-indicator">Loading animals...</div>
          ) : animals.length === 0 ? (
            <div className="text-center py-4 text-gray-500">No animals found</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {animals.map((animal) => (
                <div key={animal._id} className="border rounded-lg p-4 hover:shadow-md transition-shadow" data-testid={`animal-card-${animal.id}`}>
                  <h4 className="font-semibold text-lg">{animal.name}</h4>
                  <p className="text-gray-600 capitalize">{animal.animal_type}</p>
                  <p className="text-sm text-gray-500">{animal.breed}</p>
                  <p className="text-sm text-gray-500">Age: {animal.age}</p>
                  <span className={`inline-block px-2 py-1 text-xs rounded ${
                    animal.outcome_type === 'Available' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {animal.outcome_type}
                  </span>
                  <p className="text-xs text-gray-400 mt-2">
                    Created: {new Date(animal.created_at).toLocaleDateString()}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const VoiceNoteMDDemo = () => {
  const [voiceText, setVoiceText] = useState('');
  const [processedNote, setProcessedNote] = useState(null);
  const [medicalNotes, setMedicalNotes] = useState([]);
  const [loading, setLoading] = useState(false);

  const sampleVoiceNote = "Patient reports severe headache and nausea for the past 12 hours. Blood pressure is 160/95, pulse 88, temperature 100.1 degrees fahrenheit, respiratory rate 20. Physical examination reveals neck stiffness and photophobia. Assessment shows possible meningitis based on clinical presentation. Plan includes immediate lumbar puncture, blood cultures, and empirical antibiotic therapy with ceftriaxone.";

  const processVoiceNote = async (e) => {
    e.preventDefault();
    if (!voiceText.trim()) return;
    
    try {
      setLoading(true);
      const response = await axios.post(`${API}/voice-notes`, {
        voice_text: voiceText,
        patient_id: 'demo_patient_001',
        provider_id: 'demo_provider_001',
        note_type: 'progress_note'
      });
      
      if (response.data.ok) {
        setProcessedNote(response.data.data);
        setVoiceText('');
        fetchMedicalNotes(); // Refresh notes list
      }
    } catch (e) {
      console.error('Error processing voice note:', e);
    } finally {
      setLoading(false);
    }
  };

  const fetchMedicalNotes = async () => {
    try {
      const response = await axios.get(`${API}/voice-notes?patient_id=demo_patient_001`);
      if (response.data.ok) {
        setMedicalNotes(response.data.data);
      }
    } catch (e) {
      console.error('Error fetching medical notes:', e);
    }
  };

  const loadSampleNote = () => {
    setVoiceText(sampleVoiceNote);
  };

  useEffect(() => {
    fetchMedicalNotes();
  }, []);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">VoiceNote MD - SOAP Extraction Demo</h2>
      <p className="text-gray-600 mb-6">Convert voice notes into structured SOAP format using NLP algorithms</p>
      
      {/* Voice Input Section */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Voice Note Input</h3>
        <form onSubmit={processVoiceNote} className="space-y-4">
          <div>
            <textarea
              value={voiceText}
              onChange={(e) => setVoiceText(e.target.value)}
              placeholder="Enter voice note text here..."
              className="w-full border rounded-lg px-4 py-3 h-32 resize-none"
              data-testid="voice-text-input"
            />
            <div className="flex justify-between items-center mt-2">
              <button
                type="button"
                onClick={loadSampleNote}
                className="text-blue-500 hover:text-blue-600 text-sm"
                data-testid="load-sample-btn"
              >
                Load Sample Note
              </button>
              <span className="text-xs text-gray-500">
                Processing Target: ≤5000ms per 60-second note
              </span>
            </div>
          </div>
          <button
            type="submit"
            disabled={loading || !voiceText.trim()}
            className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg disabled:opacity-50"
            data-testid="process-note-btn"
          >
            {loading ? 'Processing...' : 'Process Voice Note'}
          </button>
        </form>
      </div>

      {/* Processed SOAP Result */}
      {processedNote && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">SOAP Extraction Result</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(processedNote.soap_sections).map(([section, data]) => (
              <div key={section} className="border rounded-lg p-4" data-testid={`soap-section-${section}`}>
                <h4 className="font-semibold capitalize text-lg mb-2 text-blue-600">{section}</h4>
                {data.sentences.length > 0 ? (
                  <div className="space-y-2">
                    {data.sentences.map((sentence, idx) => (
                      <p key={idx} className="text-gray-700 text-sm">{sentence}</p>
                    ))}
                    {data.entities && data.entities.length > 0 && (
                      <div className="mt-3">
                        <p className="text-xs font-medium text-gray-600 mb-1">Extracted Entities:</p>
                        <div className="flex flex-wrap gap-1">
                          {data.entities.map((entity, idx) => (
                            <span
                              key={idx}
                              className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded"
                            >
                              {entity.type}: {entity.value}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No content classified for this section</p>
                )}
              </div>
            ))}
          </div>
          
          {/* Processing Metadata */}
          <div className="mt-4 pt-4 border-t">
            <div className="flex flex-wrap gap-4 text-sm text-gray-600">
              <span>Processing Time: {processedNote.metadata.processing_time_ms}ms</span>
              <span>Total Sentences: {processedNote.metadata.total_sentences}</span>
              <span>Note ID: {processedNote.note_id}</span>
            </div>
          </div>
        </div>
      )}

      {/* Medical Notes History */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-semibold">Medical Notes History</h3>
          <p className="text-sm text-gray-600">Stored using enhanced repository with audit trails</p>
        </div>
        <div className="p-6">
          {medicalNotes.length === 0 ? (
            <div className="text-center py-4 text-gray-500">No medical notes found</div>
          ) : (
            <div className="space-y-4">
              {medicalNotes.slice(0, 3).map((note, idx) => (
                <div key={note._id} className="border rounded-lg p-4" data-testid={`medical-note-${idx}`}>
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium">Note #{idx + 1}</span>
                    <span className="text-xs text-gray-500">{new Date(note.processed_at).toLocaleString()}</span>
                  </div>
                  <p className="text-sm text-gray-700 mb-3 line-clamp-2">{note.original_text}</p>
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Patient ID: {note.patient_id}</span>
                    <span>Processing: {note.metadata.processing_time_ms}ms</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ComparisonDemo = () => {
  const [comparison, setComparison] = useState(null);

  const fetchComparison = async () => {
    try {
      const response = await axios.get(`${API}/demo/comparison`);
      setComparison(response.data);
    } catch (e) {
      console.error('Error fetching comparison:', e);
    }
  };

  useEffect(() => {
    fetchComparison();
  }, []);

  if (!comparison) return <div className="p-6">Loading comparison...</div>;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">CS-499 Capstone: Before vs After Comparison</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Old Approach */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-red-800 mb-4">❌ Original CRUD (CS-340)</h3>
          <div className="space-y-3">
            {Object.entries(comparison.old_approach).map(([key, value]) => (
              <div key={key} className="border-b border-red-200 pb-2">
                <span className="font-medium text-red-700 capitalize">{key.replace('_', ' ')}:</span>
                <p className="text-red-600 text-sm mt-1">{value}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Enhanced Approach */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-green-800 mb-4">✅ Enhanced Repository (CS-499)</h3>
          <div className="space-y-3">
            {Object.entries(comparison.enhanced_approach).map(([key, value]) => (
              <div key={key} className="border-b border-green-200 pb-2">
                <span className="font-medium text-green-700 capitalize">{key.replace('_', ' ')}:</span>
                <p className="text-green-600 text-sm mt-1">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Targets */}
      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-blue-800 mb-4">🎯 Performance Targets</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(comparison.performance_targets).map(([key, value]) => (
            <div key={key} className="bg-white rounded p-4 border border-blue-200">
              <span className="font-medium text-blue-700 capitalize block">{key.replace('_', ' ')}</span>
              <span className="text-blue-600 text-lg font-semibold">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const Navigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'comparison', label: 'Before vs After', icon: '🔄' },
    { id: 'animals', label: 'Animal Shelter', icon: '🐕' },
    { id: 'voicenote', label: 'VoiceNote MD', icon: '🏥' }
  ];

  return (
    <nav className="bg-white shadow-md border-b">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              data-testid={`tab-${tab.id}`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};

const Home = () => {
  const [activeTab, setActiveTab] = useState('comparison');

  const helloWorldApi = async () => {
    try {
      const response = await axios.get(`${API}/`);
      console.log('API Response:', response.data);
    } catch (e) {
      console.error(e, `errored out requesting / api`);
    }
  };

  useEffect(() => {
    helloWorldApi();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold mb-2">CS-499 Capstone Demonstration</h1>
          <p className="text-blue-100">Enhanced Repository Layer - Professional Software Engineering Practices</p>
          <p className="text-blue-200 text-sm mt-2">Milfred Martinez - Software Engineering & Design</p>
        </div>
      </header>

      {/* Navigation */}
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Content */}
      <main>
        {activeTab === 'comparison' && <ComparisonDemo />}
        {activeTab === 'animals' && <AnimalShelterDemo />}
        {activeTab === 'voicenote' && <VoiceNoteMDDemo />}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <p className="mb-2">Enhanced Repository Pattern with Safety Features & Audit Trails</p>
          <p className="text-gray-400 text-sm">
            Demonstrating professional software engineering practices for CS-499 Capstone
          </p>
        </div>
      </footer>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;