; Auto-generated. Do not edit!


(cl:in-package tiling-msg)


;//! \htmlinclude Alt.msg.html

(cl:defclass <Alt> (roslisp-msg-protocol:ros-message)
  ((alt
    :reader alt
    :initarg :alt
    :type cl:integer
    :initform 0))
)

(cl:defclass Alt (<Alt>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Alt>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Alt)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name tiling-msg:<Alt> is deprecated: use tiling-msg:Alt instead.")))

(cl:ensure-generic-function 'alt-val :lambda-list '(m))
(cl:defmethod alt-val ((m <Alt>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader tiling-msg:alt-val is deprecated.  Use tiling-msg:alt instead.")
  (alt m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Alt>) ostream)
  "Serializes a message object of type '<Alt>"
  (cl:let* ((signed (cl:slot-value msg 'alt)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Alt>) istream)
  "Deserializes a message object of type '<Alt>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'alt) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Alt>)))
  "Returns string type for a message object of type '<Alt>"
  "tiling/Alt")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Alt)))
  "Returns string type for a message object of type 'Alt"
  "tiling/Alt")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Alt>)))
  "Returns md5sum for a message object of type '<Alt>"
  "59c4254ca6636d694bc3e5b3b9a6b8c7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Alt)))
  "Returns md5sum for a message object of type 'Alt"
  "59c4254ca6636d694bc3e5b3b9a6b8c7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Alt>)))
  "Returns full string definition for message of type '<Alt>"
  (cl:format cl:nil "# Alt.msg~%int32 alt~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Alt)))
  "Returns full string definition for message of type 'Alt"
  (cl:format cl:nil "# Alt.msg~%int32 alt~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Alt>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Alt>))
  "Converts a ROS message object to a list"
  (cl:list 'Alt
    (cl:cons ':alt (alt msg))
))
