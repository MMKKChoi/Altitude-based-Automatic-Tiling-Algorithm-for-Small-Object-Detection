// Auto-generated. Do not edit!

// (in-package tiling.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Alt {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.alt = null;
    }
    else {
      if (initObj.hasOwnProperty('alt')) {
        this.alt = initObj.alt
      }
      else {
        this.alt = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Alt
    // Serialize message field [alt]
    bufferOffset = _serializer.int32(obj.alt, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Alt
    let len;
    let data = new Alt(null);
    // Deserialize message field [alt]
    data.alt = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tiling/Alt';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '59c4254ca6636d694bc3e5b3b9a6b8c7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Alt.msg
    int32 alt
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Alt(null);
    if (msg.alt !== undefined) {
      resolved.alt = msg.alt;
    }
    else {
      resolved.alt = 0
    }

    return resolved;
    }
};

module.exports = Alt;
