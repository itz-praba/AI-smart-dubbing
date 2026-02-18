import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import Layout from "./Layout";

import Login from "./components/Auth/login";
import Signup from "./components/Auth/signup";
import Forgot from "./components/Auth/Forgot";
import ForgotVerify from "./components/Auth/ForgotVerify";
import ResetPassword from "./components/Auth/ResetPassword";
import NotFound from "./components/sections/NotFound";

import Index from "./Index";
import Features from "./components/sections/Features";
import HowItWorks from "./components/sections/HowItWorks";
import Languages from "./components/sections/Languages";
import UseCases from "./components/sections/UseCases";
import Pricing from "./components/sections/Pricing";
import Testimonials from "./components/sections/Testimonials";

import { TooltipProvider } from "./components/ui/tooltip";
import { Toaster } from "./components/ui/toaster";
import { Toaster as Sonner } from "./components/ui/sonner";
import Contact from "./components/sections/Contact";
import StartDubbing from "./components/Dubbing/StartDubbing";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />

        <BrowserRouter>
          <Routes>
            {/* Layout wrapper */}
            <Route element={<Layout />}>
              <Route path="/" element={
                  <Index />
                } />

              {/* Auth */}
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
              <Route path="/forgot" element={<Forgot />} />
              <Route path="/forgot/verify" element={<ForgotVerify />} />
              <Route path="/forgot/reset" element={<ResetPassword />} />

              {/* Marketing */}
              <Route path="/features" element={
                <>
                <Features />
                <UseCases />
                <Testimonials />
                </>} />
              <Route path="/how-it-works" element={<HowItWorks />} />
              <Route path="/languages" element={<Languages />} />
              <Route path="/use-cases" element={<UseCases />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/testimonials" element={<Testimonials />} />
              <Route path="/contact" element={<Contact />} />
            {/*Dubbing */}

            <Route path="/start-dubbing" element={<StartDubbing />} /> 
            </Route>


            {/* 404 (still uses layout) */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
