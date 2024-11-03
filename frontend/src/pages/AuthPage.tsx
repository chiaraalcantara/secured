import React, { useState } from "react";
import Signup from "../components/Signup";
import { Link } from "react-router-dom";

const AuthPage = () => {
  const [isSignup, setIsSignup] = useState(true);

  const toggleSlider = () => setIsSignup(!isSignup);

  return (
    <div className="h-screen p-8 flex">
      {/* Left */}
      <div className="flex-1 bg-primary-500 rounded-2xl flex flex-col items-center justify-center text-center text-white">
        <h1 className="text-2xl font-bold mb-2">Welcome to Secured+</h1>
        <p className="text-lg">Built with passion by{' '}
          <a href="https://watai.ca" target="_blank" rel="noopener noreferrer" className="text-yellow-400 hover:text-yellow-200 transition-colors duration-300">
            WAT.ai
          </a>
        </p>

        <img
          src={"secured.webp"}
          alt="Secured+ Logo"
          className="h-60 w-auto mt-24 mb-24"
        />

        <h2 className="text-2xl font-semibold mb-2">Seamless Management</h2>
        <p className="text-lg">
          Effortlessly manage your students in real-time.
        </p>
      </div>

      {/* Right */}
      <div className="flex-1 flex items-center justify-center">
        <div className="p-6 w-7/12 h-3/4">
        <div className="flex items-center mb-6">
          <img
            src={"securedPrimary.webp"}
            alt="Secured+ Logo"
            className="h-8 w-8 mr-2"
          />
          <h1 className="text-2xl">Secured+</h1>
        </div>
        
          {/* Slider Toggle */}
          <div className="relative w-full bg-secondary-500 rounded-lg p-2 mb-6">
          <div
            className={`absolute top-2 bottom-2  w-[calc(50%-8px)] transition-transform duration-300 ${
              isSignup ? "translate-x-0 bg-primary-500" : "translate-x-full bg-primary-500"
            } rounded-lg`}
          ></div>
            <div className="relative z-10 flex justify-between">
              <button
                className={`w-1/2 text-center py-2 ${
                  isSignup ? "text-white" : "text-gray-400"
                }`}
                onClick={() => setIsSignup(true)}
              >
                Sign Up
              </button>
              <button
                className={`w-1/2 text-center py-2 ${
                  isSignup ? "text-gray-400" : "text-white"
                }`}
                onClick={() => setIsSignup(false)}
              >
                Sign In
              </button>
            </div>
          </div>
          
          {/* Forms */}
          {isSignup && <Signup />}
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
