import Swal from "sweetalert2";


const darkAlert = Swal.mixin({
  background: "#0f172a", // dark background
  color: "#e5e7eb",      // light text
  confirmButtonColor: "#0ea5e9", // match your button color
  cancelButtonColor: "#334155",
  iconColor: "#ef4444", // red error icon
  customClass: {
    popup: "dark-swal-popup",
    title: "dark-swal-title",
    confirmButton: "dark-swal-btn",
  },
});

export default darkAlert;
