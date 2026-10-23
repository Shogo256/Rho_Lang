// SPDX-License-Identifier: Apache-2.0
//! Shared libraries opened with the C library's dlopen, called directly:
//! how `rhoc --run` and difftest load the kernels they compile.

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

fn last_error() -> String {
    // SAFETY: dlerror's string lives until the next dl call on this thread;
    // it is copied at once.
    let e = unsafe { dlerror() };
    if e.is_null() {
        "unknown dlopen error".to_string()
    } else {
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

/// An open shared library, closed when dropped.
pub struct Library(*mut c_void);

impl Library {
    pub fn open(path: &Path) -> Result<Library, String> {
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: a NUL-terminated path.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            Err(last_error())
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
        let s = dlsym(self.0, c.as_ptr());
        if s.is_null() {
            Err(format!("{name}: {}", last_error()))
        } else {
            Ok(std::mem::transmute_copy::<*mut c_void, T>(&s))
        }
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle dlopen gave, closed once.
        unsafe { dlclose(self.0) };
    }
}
