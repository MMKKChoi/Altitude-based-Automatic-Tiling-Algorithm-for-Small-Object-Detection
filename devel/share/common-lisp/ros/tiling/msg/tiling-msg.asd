
(cl:in-package :asdf)

(defsystem "tiling-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Alt" :depends-on ("_package_Alt"))
    (:file "_package_Alt" :depends-on ("_package"))
  ))