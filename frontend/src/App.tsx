import { Routes, Route, Navigate } from 'react-router-dom';
import { SignedIn, SignedOut } from '@clerk/clerk-react';
import AuthPage from './pages/AuthPage';
import Setup from './pages/SetupPage';
import HomePage from './pages/HomePage'; // New HomePage component
import FaceDetectionPage from './pages/FaceDetectionPage';
function App() {
  return (
    <header>
      <SignedOut>
        <Routes>
          <Route path="/" element={<FaceDetectionPage />} />
        </Routes>
      </SignedOut>
      <SignedIn>
        <Routes>
          <Route path="/setup" element={<Setup />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="*" element={<Navigate to="/home" />} />
        </Routes>
      </SignedIn>
    </header>
  );
}

export default App;