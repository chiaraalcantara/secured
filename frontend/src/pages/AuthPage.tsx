

const AuthPage = () => {
  return (
    <div className="h-screen p-8 flex space-x-4">
      {/* Left */}
      <div className="flex-1 bg-primary rounded-2xl flex flex-col items-center justify-center text-center text-white">
        <h1 className="text-2xl font-bold mb-2">Welcome to Secured+</h1>
        <p className="text-sm mb-8">Built with passion by WAT.ai</p>

        <img src={"secured.webp"} alt="Lock Icon" className="h-80 w-auto mb-8" />

        <h2 className="text-2xl font-semibold mb-2">Seamless Management</h2>
        <p className="text-sm">Effortlessly manage your students in real-time.</p>
      </div>

      {/* Right */}
      <div className="flex-1 flex items-center justify-center">
        <div className="bg-secondary p-6 w-96 h-3/4"> {/* Container-like div */}
          <h2 className="text-xl mb-4">Login</h2>
          <form>
            <div className="mb-4">
              <label htmlFor="email" className="block text-sm font-medium">Email</label>
              <input type="email" id="email" className="mt-1 block w-full border border-gray-300 rounded-md p-2" required />
            </div>
            <div className="mb-4">
              <label htmlFor="password" className="block text-sm font-medium">Password</label>
              <input type="password" id="password" className="mt-1 block w-full border border-gray-300 rounded-md p-2" required />
            </div>
            <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded-md">Login</button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default AuthPage;