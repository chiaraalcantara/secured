import { useState } from "react";
import CustomizeClass from "../components/CustomizeClass";

const Setup = () => {
  const [step, setStep] = useState(1);
  
  return (
    <div className="h-screen p-8 flex bg-secondary-500">
      <div className="w-full h-full bg-white rounded-2xl flex flex-col items-center justify-start">
        <div className="flex items-center mb-10 mt-10">
          <img
            src={"securedPrimary.webp"}
            alt="Secured+ Logo"
            className="h-8 w-8 mr-2"
          />
          <h1 className="text-2xl">Secured+</h1>
        </div>

        <p className="text-xl text-gray-400 mb-2">{step}/2</p>
        
        <CustomizeClass />
      </div>
    </div>
  );
};

export default Setup;