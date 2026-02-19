export const checkSession = async () => {
  try {
    const res = await fetch("http://localhost:8001/me", {
      method: "GET",
      credentials: "include", // 🔥 VERY IMPORTANT
    });

    if (!res.ok) return false;

    const data = await res.json();
    return data.authenticated === true;
  } catch (err) {
    return false;
  }
};
