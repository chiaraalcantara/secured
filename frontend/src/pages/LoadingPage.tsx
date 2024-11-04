import { motion } from "framer-motion";
import { useEffect } from "react";

interface LoadingPageProps {
  onLoadingComplete: () => void;
}

const LoadingPage = ({ onLoadingComplete }: LoadingPageProps) => {
  const text = "Secured+";

  // Container animation
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.5,
        delay: 0.5, // 0.5-second delay before fade-in starts
        when: "beforeChildren",
        staggerChildren: 0.1,
      },
    },
    exit: {
      opacity: 0,
      transition: {
        duration: 0.5,
        when: "afterChildren",
        staggerChildren: 0.05,
        staggerDirection: -1,
      },
    },
  };

  // Letter animation
  const letterVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.2,
      },
    },
    exit: {
      opacity: 0,
      y: -30,
      transition: {
        duration: 0.2,
      },
    },
  };

  // Handle the loading sequence
  useEffect(() => {
    const timer = setTimeout(() => {
      onLoadingComplete();
    }, 3000); // 3 seconds for loading animation

    return () => clearTimeout(timer);
  }, [onLoadingComplete]);

  return (
    <motion.div
      key="loading"
      className="h-screen w-screen flex items-center justify-center bg-white"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      exit="exit"
    >
      <div className="flex">
        {text.split("").map((char, index) => (
          <motion.span
            key={index}
            className="text-6xl text-primary-500"
            variants={letterVariants}
          >
            {char}
          </motion.span>
        ))}
      </div>
    </motion.div>
  );
};

export default LoadingPage;