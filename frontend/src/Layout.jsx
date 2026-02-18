import { Outlet, useLocation } from "react-router-dom";
import Navbar from "./components/sections/Navbar";
import Footer from "./components/sections/Footer";

const Layout = () => {
  const location = useLocation();

  const authRoutes = [
    "/login",
    "/signup",
    "/forgot",
    "/forgot/verify",
    "/forgot/reset",
  ];

  const showNav = !authRoutes.some(route =>
    location.pathname.startsWith(route)
  );

  return (
    <main className="min-h-screen bg-background flex flex-col">
      {showNav && <Navbar />}

      <div className="flex-1 w-full">
        <Outlet />
      </div>

      {showNav && <Footer />}
    </main>
  );
};

export default Layout;
