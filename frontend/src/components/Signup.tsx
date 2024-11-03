import React, { useState } from "react";

const Signup: React.FC = () => {
  const [password, setPassword] = useState("");

  // Helper function to check password criteria
  const isWeakPassword = password.length < 8;
  const containsNameOrEmail = /name|email/i.test(password); // Replace with actual checks for user name or email
  const hasMinLength = password.length >= 8;
  const hasNumberOrSymbol = /[0-9!@#$%^&*]/.test(password);

  return (
    <div className="w-full">
      {/* Classroom ID */}
      <div className="mb-4">
        <label className="block text-gray-700 font-bold mb-2" htmlFor="classroomID">
          Classroom ID
        </label>
        <input
          type="text"
          id="classroomID"
          className="w-full p-2 border-2 border-secondary-500 rounded-lg"
        />
      </div>

      {/* Password */}
      <div className="mb-4">
        <div className="flex justify-between items-center">
          <label className="text-gray-700 font-bold" htmlFor="password">
            Password
          </label>
          <a href="#" className="text-gray-400 text-sm">
            Forgot password?
          </a>
        </div>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-2 border-2 border-secondary-500 rounded-lg"
        />
      </div>

      {/* Password Strength Check */}
      <div className="mb-8 space-y-1">
        <div className={`text-sm ${isWeakPassword ? "text-red-500" : "text-green-500"}`}>
          Password strength: {isWeakPassword ? "Weak" : "Strong"}
        </div>
        <div className={`text-sm ${containsNameOrEmail ? "text-red-500" : "text-green-500"}`}>
          Cannot contain your name or email address
        </div>
        <div className={`text-sm ${hasMinLength ? "text-green-500" : "text-red-500"}`}>
          At least 8 characters
        </div>
        <div className={`text-sm ${hasNumberOrSymbol ? "text-green-500" : "text-red-500"}`}>
          Contains a number or symbol
        </div>
      </div>

      <button className="bg-primary-500 w-full text-white mb-4 py-3 rounded-lg hover:bg-secondary-900 transition-colors duration-300">Create Classroom</button>

       {/* OR Divider */}
       <div className="flex items-center justify-center">
        <div className="w-1/3 border-t border-gray-200"></div>
        <span className="mx-2 text-sm text-gray-300">OR</span>
        <div className="w-1/3 border-t border-gray-200"></div>
      </div>
    </div>
  );
};

export default Signup;
