import React, { useState } from "react";
import { motion } from "framer-motion";

interface AuthFormProps {
  isSignup: boolean;
}

const AuthForm: React.FC<AuthFormProps> = ({ isSignup }) => {
  const [password, setPassword] = useState("");

  const isWeakPassword = password.length < 8;
  const containsNameOrEmail = /name|email/i.test(password);
  const hasMinLength = password.length >= 8;
  const hasNumberOrSymbol = /[0-9!@#$%^&*]/.test(password);

  const wiggleAnimation = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1, rotate: [0, 15, -15, 10, -10, 5, -5, 0] },
  };

  return (
    <div className="w-full">
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

      <div className={`mb-${isSignup ? "2" : "8"}`}>
        <div className="flex justify-between items-center">
          <label className="text-gray-700 font-bold mb-2" htmlFor="password">
            Password
          </label>
          {!isSignup && (
            <a href="#" className="text-gray-400 text-sm">
              Forgot password?
            </a>
          )}
        </div>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-2 border-2 border-secondary-500 rounded-lg"
        />
      </div>

      {isSignup && (
        <div className="mb-8 space-y-1">
          <div className="flex items-center text-sm">
            <motion.span
              className={`mr-2 ${isWeakPassword ? "text-secondary-600" : "text-green-500"}`}
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={wiggleAnimation}
              key={isWeakPassword ? "weak" : "strong"}
            >
              {isWeakPassword ? "✖" : "✔"}
            </motion.span>
            <span className={`${isWeakPassword ? "text-secondary-600" : "text-green-500"}`}>
              Password strength: {isWeakPassword ? "Weak" : "Strong"}
            </span>
          </div>

          <div className="flex items-center text-sm">
            <motion.span
              className={`mr-2 ${containsNameOrEmail ? "text-secondary-600" : "text-green-500"}`}
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={wiggleAnimation}
              key={containsNameOrEmail ? "contains" : "no-contains"}
            >
              {containsNameOrEmail ? "✖" : "✔"}
            </motion.span>
            <span className={`${containsNameOrEmail ? "text-secondary-600" : "text-green-500"}`}>
              Cannot contain your name or email address
            </span>
          </div>

          <div className="flex items-center text-sm">
            <motion.span
              className={`mr-2 ${hasMinLength ? "text-green-500" : "text-secondary-600"}`}
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={wiggleAnimation}
              key={hasMinLength ? "min-length" : "short-length"}
            >
              {hasMinLength ? "✔" : "✖"}
            </motion.span>
            <span className={`${hasMinLength ? "text-green-500" : "text-secondary-600"}`}>
              At least 8 characters
            </span>
          </div>

          <div className="flex items-center text-sm">
            <motion.span
              className={`mr-2 ${hasNumberOrSymbol ? "text-green-500" : "text-secondary-600"}`}
              initial="hidden"
              animate="visible"
              exit="hidden"
              variants={wiggleAnimation}
              key={hasNumberOrSymbol ? "has-symbol" : "no-symbol"}
            >
              {hasNumberOrSymbol ? "✔" : "✖"}
            </motion.span>
            <span className={`${hasNumberOrSymbol ? "text-green-500" : "text-secondary-600"}`}>
              Contains a number or symbol
            </span>
          </div>
        </div>
      )}

      <button className="bg-primary-500 w-full text-white mb-4 py-3 rounded-lg hover:bg-secondary-900 transition-colors duration-300">
        {isSignup ? "Create Classroom" : "Sign In"}
      </button>

      <div className="flex items-center justify-center">
        <div className="w-1/4 border-t border-secondary-600"></div>
        <span className="mx-2 text-sm text-secondary-600">OR</span>
        <div className="w-1/4 border-t border-secondary-600"></div>
      </div>
    </div>
  );
};

export default AuthForm;