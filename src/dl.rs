// SPDX-License-Identifier: Apache-2.0
//! Shared libraries opened by the operating system's loader, called
//! directly: dlopen on Unix, LoadLibraryW on Windows. This is how
//! `rhoc --run` and difftest load the kernels they compile.

use std::ffi::{c_void, CString};
use std::path::Path;

#[cfg(unix)]
mod sys {
    use std::ffi::{c_char, c_int, c_void, CStr, CString};
    use std::path::Path;

    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn dlerror() -> *const c_char;
    }

    const RTLD_NOW: c_int = 2;

    pub fn last_error() -> String {
        // SAFETY: dlerror's string lives until the next dl call on this
        // thread; it is copied at once.
        let e = unsafe { dlerror() };
        if e.is_null() {
            "unknown dlopen error".to_string()
        } else {
            unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
        }
    }

    pub fn open(path: &Path) -> Result<*mut c_void, String> {
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: a NUL-terminated path.
        Ok(unsafe { dlopen(c.as_ptr(), RTLD_NOW) })
    }

    pub unsafe fn symbol(handle: *mut c_void, name: &CString) -> *mut c_void {
        dlsym(handle, name.as_ptr())
    }

    pub unsafe fn close(handle: *mut c_void) {
        dlclose(handle);
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::{c_char, c_void, CString};
    use std::path::Path;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    pub fn last_error() -> String {
        // SAFETY: reads the calling thread's last error code.
        format!("Windows error {}", unsafe { GetLastError() })
    }

    pub fn open(path: &Path) -> Result<*mut c_void, String> {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        if wide[..wide.len() - 1].contains(&0) {
            return Err("path contains a NUL character".to_string());
        }
        // SAFETY: a NUL-terminated UTF-16 path.
        Ok(unsafe { LoadLibraryW(wide.as_ptr()) })
    }

    pub unsafe fn symbol(handle: *mut c_void, name: &CString) -> *mut c_void {
        GetProcAddress(handle, name.as_ptr())
    }

    pub unsafe fn close(handle: *mut c_void) {
        FreeLibrary(handle);
    }
}

/// An open shared library, closed when dropped.
pub struct Library(*mut c_void);

impl Library {
    pub fn open(path: &Path) -> Result<Library, String> {
        let h = sys::open(path)?;
        if h.is_null() {
            Err(sys::last_error())
        } else {
            Ok(Library(h))
        }
    }

    /// The symbol `name` as a `T`, a function pointer of the symbol's type.
    ///
    /// # Safety
    /// `T` must be the type of what the symbol is, and must not outlive the
    /// library.
    pub unsafe fn get<T: Copy>(&self, name: &str) -> Result<T, String> {
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        let c = CString::new(name).map_err(|e| e.to_string())?;
        let s = sys::symbol(self.0, &c);
        if s.is_null() {
            Err(format!("{name}: {}", sys::last_error()))
        } else {
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&s))
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle the loader gave, closed once.
        unsafe { sys::close(self.0) };
    }
}
