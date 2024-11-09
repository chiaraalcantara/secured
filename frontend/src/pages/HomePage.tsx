import { SignOutButton } from "@clerk/clerk-react";

const HomePage = () => {
  return (
    <div className="h-screen flex items-center justify-center bg-white">
      <h1 className="text-4xl text-primary-500">Welcome to the Home Page!</h1>
      <SignOutButton />
    </div>
  );
};

export default HomePage;