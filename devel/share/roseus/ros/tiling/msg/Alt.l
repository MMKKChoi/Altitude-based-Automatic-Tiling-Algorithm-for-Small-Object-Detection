;; Auto-generated. Do not edit!


(when (boundp 'tiling::Alt)
  (if (not (find-package "TILING"))
    (make-package "TILING"))
  (shadow 'Alt (find-package "TILING")))
(unless (find-package "TILING::ALT")
  (make-package "TILING::ALT"))

(in-package "ROS")
;;//! \htmlinclude Alt.msg.html


(defclass tiling::Alt
  :super ros::object
  :slots (_alt ))

(defmethod tiling::Alt
  (:init
   (&key
    ((:alt __alt) 0)
    )
   (send-super :init)
   (setq _alt (round __alt))
   self)
  (:alt
   (&optional __alt)
   (if __alt (setq _alt __alt)) _alt)
  (:serialization-length
   ()
   (+
    ;; int32 _alt
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _alt
       (write-long _alt s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _alt
     (setq _alt (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get tiling::Alt :md5sum-) "59c4254ca6636d694bc3e5b3b9a6b8c7")
(setf (get tiling::Alt :datatype-) "tiling/Alt")
(setf (get tiling::Alt :definition-)
      "# Alt.msg
int32 alt

")



(provide :tiling/Alt "59c4254ca6636d694bc3e5b3b9a6b8c7")


