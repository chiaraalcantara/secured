import React, { useState } from 'react';

const FaceDetectionPage = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [facesFound, setFacesFound] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0]);
      setResultImage(null);
      setFacesFound(0);
    }
  };

  const detectFaces = async () => {
    if (!selectedImage) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/detect-faces', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setResultImage(`data:image/jpeg;base64,${data.image}`);
      setFacesFound(data.faces_found);
    } catch (error) {
      console.error('Error detecting faces:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Face Detection</h1>
      
      <div className="mb-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="mb-2"
        />
        <button
          onClick={detectFaces}
          disabled={!selectedImage || loading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
        >
          {loading ? 'Processing...' : 'Detect Faces'}
        </button>
      </div>

      {resultImage && (
        <div>
          <p className="mb-2">Found {facesFound} faces</p>
          <img
            src={resultImage}
            alt="Result"
            className="max-w-full h-auto border border-gray-300"
          />
        </div>
      )}
    </div>
  );
};

export default FaceDetectionPage;