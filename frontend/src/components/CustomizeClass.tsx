const CustomizeClass = () => {
  return (
    <div className="w-full h-full bg-white rounded-2xl flex flex-col items-center">
      <h1 className="text-3xl font-medium mb-2">Customize your Classroom</h1>
      <p className="text-gray-500 text-lg mb-8">Setup your Classroom for students that may join later.</p>
      
      <div className="w-full max-w-md px-4">
        <div className="mb-6">
          <label className="block mb-2">
            Classroom Name <span className="text-red-500">*</span>
          </label>
          <input 
            type="text"
            className="w-full p-2 border-2 border-secondary-500 rounded-lg"
            placeholder="e.g. MAT 137"
          />
        </div>

        <div className="mb-6">
          <label className="block mb-2">
            Subject <span className="text-red-500">*</span>
          </label>
          <select className="w-full p-2 border-2 border-secondary-500 rounded-lg"> 
            <option>Math</option>
            <option>Science</option>
            <option>English</option>
            <option>History</option>
          </select>
        </div>

        <div className="mt-20">
          <button className="bg-primary-500 w-full text-white mb-4 py-3 rounded-lg hover:bg-secondary-900 transition-colors duration-300">
            Continue
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomizeClass;