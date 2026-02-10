import { Routes, Route } from 'react-router-dom'
import Login from './components/Auth/login'
import Signup from './components/Auth/signup';
import Forgot from './components/Auth/Forgot';
import ForgotVerify from './components/Auth/ForgotVerify';
import ResetPassword from './components/Auth/ResetPassword';

function App() {
  return (
    <Routes>
      <Route path='/' element={<p>Home page</p>} />
      <Route path='/login' element={<Login />} />
      <Route path='/forgot' element={<Forgot />} />
      <Route path='/forgot/verify' element={<ForgotVerify />} />
      <Route path='/forgot/reset' element={<ResetPassword />} />
      <Route path='/signup' element={<Signup />} />
    </Routes>
  )
}

export default App;
