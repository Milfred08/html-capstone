# CS-499 Capstone Project

**Enhanced Repository Layer - Professional Software Engineering Practices**

**Author:** Milfred Martinez

---

## 📋 Project Overview

This capstone project demonstrates the transformation of a basic Python CRUD script (originally developed for CS-340) into a professional, production-ready full-stack application. The project showcases advanced software engineering practices including repository patterns, algorithm optimization, and database management through three major enhancement milestones.

### Key Achievements

- ✅ **Repository Pattern Implementation** - Professional data access layer with validation and audit trails
- ✅ **Algorithm Optimization** - Multiple NLP algorithms with performance benchmarking
- ✅ **Advanced Database Management** - Aggregation pipelines, indexing, and analytics
- ✅ **Full-Stack Architecture** - React frontend, FastAPI backend, MongoDB database
- ✅ **Comprehensive Testing** - Unit tests, integration tests, and performance metrics

---

## 🎯 Capstone Milestones

### Milestone 2: Software Design & Engineering
**Enhancement:** Repository Pattern Refactoring

Transformed the basic CRUD module into a professional repository layer:
- Input validation and type checking
- Structured error handling with deterministic responses
- Safe delete operations with confirmation requirements
- Audit trails and logging capabilities
- Environment-based configuration management
- Index helpers for performance optimization

**Files:**
- `backend/AnimalShelter_enhanced.py` - Enhanced repository for animal data
- `backend/MedicalNotes_repository.py` - Repository for medical notes
- `backend/test_animal_shelter.py` - Comprehensive test suite

### Milestone 3: Algorithms & Data Structures
**Enhancement:** NLP Pipeline with Advanced Algorithms

Implemented multiple algorithm approaches for medical text processing:
- **Hybrid Algorithm** - Combines pattern matching with rule-based classification
- **Trie-based Search** - Efficient prefix-tree for entity extraction
- **Hash-based Lookup** - O(1) average case for keyword matching
- **Rule-based Classification** - Medical domain-specific rules

**Performance Target:** ≤5000ms processing time for 60-second voice notes

**Files:**
- `backend/nlp_soap.py` - NLP pipeline with multiple algorithms
- `backend/test_nlp_soap_enhanced.py` - Algorithm benchmarking tests

### Milestone 4: Databases
**Enhancement:** Advanced Database Management

Implemented sophisticated database features:
- Aggregation pipelines for complex analytics
- Strategic indexing for query optimization
- Real-time performance monitoring
- Connection pooling and resource management
- Business intelligence capabilities
- Data integrity enforcement

**Files:**
- `backend/AdvancedDatabaseManager.py` - Database management and analytics
- `backend/test_advanced_database.py` - Database feature tests

---

## 🛠️ Technology Stack

### Frontend
- **React.js** - Component-based UI framework
- **TailwindCSS** - Utility-first CSS framework
- **Axios** - HTTP client for API communication
- **React Router** - Client-side routing

### Backend
- **FastAPI** - Modern Python web framework with async support
- **Motor** - Async MongoDB driver
- **PyMongo** - MongoDB Python driver
- **Pydantic** - Data validation using Python type annotations

### Database
- **MongoDB** - NoSQL document database
- **Indexing** - Strategic indexes for performance
- **Aggregation** - Complex data analysis pipelines

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 14+** and **Yarn**
- **MongoDB 4.4+** (local installation or MongoDB Atlas account)

### Installation

#### 1. Clone the Repository

```bash
git clone
cd cs499_capstone
```

#### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and configure:
# - MONGO_URL: Your MongoDB connection string
# - DB_NAME: Database name (default: cs499_capstone)
# - CORS_ORIGINS: Frontend URL (default: http://localhost:3000)
```

#### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
yarn install

# Create environment file
cp .env.example .env

# Edit .env and set:
# REACT_APP_BACKEND_URL=http://localhost:8001
```

#### 4. Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
yarn start
```

The application will be available at:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8001
- **API Documentation:** http://localhost:8001/docs

---

## 📁 Project Structure

```
cs499_capstone/
├── backend/
│   ├── AdvancedDatabaseManager.py       # Database analytics and management
│   ├── AnimalShelter_enhanced.py        # Enhanced repository pattern
│   ├── AnimalShelter_old.py             # Original CRUD (for comparison)
│   ├── MedicalNotes_repository.py       # Medical notes repository
│   ├── nlp_soap.py                      # NLP pipeline with algorithms
│   ├── server.py                        # FastAPI application
│   ├── requirements.txt                 # Python dependencies
│   ├── .env.example                     # Environment configuration template
│   ├── test_*.py                        # Test suites
│   └── capstone_demo.py                 # Demonstration scripts
│
├── frontend/
│   ├── src/
│   │   ├── App.js                       # Main React application
│   │   ├── App.css                      # Application styles
│   │   ├── index.js                     # React entry point
│   │   └── components/                  # Reusable UI components
│   ├── public/
│   │   └── index.html                   # HTML template
│   ├── package.json                     # Node.js dependencies
│   └── .env.example                     # Environment configuration template
│
├── docs/
│   ├── index.html                       # GitHub Pages documentation
│   └── screenshots/                     # Application screenshots
│
└── README.md                            # This file
```

---

## 🎯 Features

### 1. Animal Shelter Management System

Interactive CRUD application demonstrating enhanced repository pattern:

- **Create:** Add new animals with validation
- **Read:** View all animals with pagination support
- **Update:** Modify animal records safely
- **Delete:** Safe deletion with confirmation
- **Audit Trail:** Track all operations with timestamps

**Technologies:** Repository pattern, input validation, error handling

### 2. VoiceNote MD - Medical Documentation

NLP-powered medical note processing:

- **Voice-to-Text Processing:** Convert medical dictations to structured notes
- **SOAP Extraction:** Automatic classification into Subjective, Objective, Assessment, Plan
- **Entity Recognition:** Extract medical entities (symptoms, diagnoses, medications)
- **Performance Metrics:** Real-time processing time tracking
- **Note History:** View and search previous medical notes

**Technologies:** NLP algorithms, pattern matching, entity extraction

### 3. Before vs After Comparison

Side-by-side comparison demonstrating improvements:

- **Return Values:** Inconsistent vs Consistent structured responses
- **Error Handling:** Basic vs Comprehensive error management
- **Validation:** None vs Type checking and required fields
- **Safety:** No protection vs Confirmation-based safe operations
- **Audit:** No tracking vs Full audit trail
- **Configuration:** Hardcoded vs Environment-based

---

## 🧪 Testing

### Run All Tests

```bash
cd backend
pytest test_*.py -v
```

### Run Specific Test Suites

```bash
# Repository pattern tests
pytest test_animal_shelter.py -v

# NLP algorithm tests
pytest test_nlp_soap_enhanced.py -v

# Database management tests
pytest test_advanced_database.py -v
```

### Test Coverage

- **Unit Tests:** Individual function and method testing
- **Integration Tests:** End-to-end workflow testing
- **Performance Tests:** Algorithm benchmarking and optimization
- **Edge Cases:** Boundary conditions and error scenarios

---

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| NLP Processing (60s note) | ≤5000ms | ✅ Achieved |
| Database Query (indexed) | <100ms | ✅ Achieved |
| API Response Time | <200ms avg | ✅ Achieved |
| Repository Operations | <50ms | ✅ Achieved |

---

## 🔧 Configuration

### Backend Environment Variables (.env)

```env
# MongoDB Configuration
MONGO_URL=mongodb://localhost:27017
DB_NAME=cs499_capstone

# Server Configuration
CORS_ORIGINS=http://localhost:3000
```

### Frontend Environment Variables (.env)

```env
# Backend API URL
REACT_APP_BACKEND_URL=http://localhost:8001
```

---

## 📚 Documentation

### Code Documentation

All modules include comprehensive docstrings:
- Function purposes and parameters
- Return value specifications
- Usage examples
- Error handling notes

### API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI:** http://localhost:8001/docs
- **ReDoc:** http://localhost:8001/redoc

### GitHub Pages

Project showcase and portfolio: [View Documentation](docs/index.html)

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **Software Design & Architecture**
   - Repository pattern implementation
   - Separation of concerns
   - SOLID principles application

2. **Algorithms & Data Structures**
   - Multiple algorithm implementations
   - Complexity analysis (Big O notation)
   - Performance optimization strategies

3. **Database Management**
   - Schema design and indexing
   - Aggregation pipelines
   - Query optimization

4. **Full-Stack Development**
   - RESTful API design
   - Asynchronous programming
   - State management in React

5. **Software Quality**
   - Comprehensive testing
   - Error handling and validation
   - Code documentation

6. **Professional Practices**
   - Environment-based configuration
   - Security best practices
   - Code maintainability

---

## 🔒 Security Considerations

- ✅ Environment variables for sensitive data
- ✅ Input validation on all user inputs
- ✅ CORS configuration for API security
- ✅ Safe delete operations with confirmations
- ✅ Error messages without sensitive information
- ✅ MongoDB connection string security

---

## 📝 License

This project is submitted as part of CS-499 Capstone coursework at Southern New Hampshire University.

---

## 👤 Author

**Milfred Martinez**  
Computer Science - Software Engineering  
Southern New Hampshire University

---

## 🙏 Acknowledgments

- Southern New Hampshire University - CS-499 Capstone Course
- FastAPI and React.js communities for excellent documentation
- MongoDB documentation for database optimization techniques

