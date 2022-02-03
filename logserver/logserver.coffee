{WebSocketServer} = require 'ws'
uuid = require 'uuid'
fs = require 'fs'

# TODO:
# SSL
# Configurable port
# Session info from request
# Error handling

server = new WebSocketServer {port: 8082}

server.on "connection", (ws, request) ->
	session_id = uuid.v1()
	path = "logs/" + session_id
	fs.open path, "a", (err, fd) ->
		ws.on "message", (data) ->
			fs.write fd, data, ->
		ws.on "close", ->
			fs.close fd, ->
