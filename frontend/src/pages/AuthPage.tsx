import { useState } from "react";
import Signup from "../components/Signup";
import lockImage from "../assets/lock.png";
import WATaiLogo from "../assets/WATaiLogo.svg";
import { motion, AnimatePresence } from "framer-motion";
import LoadingPage from "./LoadingPage";

const AuthPage = () => {
  const [isSignup, setIsSignup] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  // Content animation
  const contentVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { duration: 0.5 },
    },
  };

  return (
    <AnimatePresence mode="wait">
      {isLoading ? (
        <LoadingPage onLoadingComplete={() => setIsLoading(false)} />
      ) : (
        // Main Content
        <motion.div
          key="content"
          className="h-screen p-8 flex"
          variants={contentVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Left - Hidden on mobile, visible on md screens and up */}
          <div className="hidden md:flex flex-1 bg-primary-500 rounded-2xl flex-col items-center justify-center text-center text-white">
            <h1 className="text-3xl font-bold mb-2">Welcome to Secured+</h1>
            <p className="text-lg">
              Built with passion by{" "}
              <a
                href="https://watai.ca"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-block"
              >
                <img
                  src={WATaiLogo}
                  alt="WAT.ai Logo"
                  className="h-6 inline -mt-2 transition-all duration-300 hover:brightness-125"
                />
              </a>
            </p>

            <img
              src={lockImage}
              alt="Lock Icon"
              className="h-48 w-auto mt-24 mb-24"
            />

            <h2 className="text-3xl font-semibold mb-2">Seamless Management</h2>
            <p className="text-lg">
              Effortlessly manage your students in real-time.
            </p>
          </div>

          {/* Right - Always visible, full width on mobile */}
          <div className="flex-1 flex items-center justify-center">
            <div className="p-6 w-full md:w-7/12 h-3/4">
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
                  className={`absolute top-2 bottom-2 w-[calc(50%-8px)] transition-transform duration-300 ${
                    isSignup
                      ? "translate-x-0 bg-primary-500"
                      : "translate-x-full bg-primary-500"
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
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AuthPage;
