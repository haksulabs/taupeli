{WebSocketServer} = require 'ws'
uuid = require 'uuid'
fs = require 'fs'
{promisify} = require 'util'
http = require "http"

# TODO:
# SSL
# Configurable port
# More session info from request?

isValidName = (name) ->
	return /^[ \w_-]*$/.test name

do ->
	server = http.createServer()
	wss = new WebSocketServer {server: server}
	
	wss.on "connection", (ws, request) ->
		session_id = uuid.v1()
		url = new URL request.url, "ws://example.com"
		session_name = url.searchParams.get("name") ? ""
		session_id = session_name + "-" + session_id
		if not isValidName session_id
			ws.close(1011, JSON.stringify {error: "Invalid session name"})
			return
		
		path = "logs/" + session_id
		
		try
			fd = await promisify(fs.open)(path, "a")
		catch err
			console.error "Failed to open logfile #{path}: #{err}"
			ws.close(1011, JSON.stringify {error: "Opening logfile failed"})
			return
		ws.on "message", (data) ->
			try
				# TODO: Can we have a race condition here?"
				await promisify(fs.write) fd, data + "\n"
			catch err
				console.error "Writing to logfile #{path} failed: #{err}"
				ws.send JSON.stringify {error: "Failed to write logfile"}
		
		ws.on "close", ->
			await promisify(fs.close) fd
	
	
	server.on "request", (req, res) ->
		url = new URL req.url, "ws://example.com"
		if req.method == "GET"
			session_name = url.searchParams.get("name") ? ""
			session_id = session_name + "-" + uuid.v1()
			if not isValidName session_id
				res.writeHead 400
				res.end JSON.stringify {error: "Invalid session name"}
				return
			res.writeHead 200
			res.end JSON.stringify {session_id: session_id}
		
		if req.method != "POST"
			res.writeHead 405
			res.end JSON.stringify {error: "Invalid method"}

		session_id = url.searchParams.get("session_id") ? ""
		if not session_id or not isValidName session_id
			res.writeHead 400
			res.end JSON.stringify {error: "Invalid session id"}
			return
		
		path = "logs/" + session_id
		
		data = ""
		req.on "data", (d) -> data += d
		req.on "end", ->
			try
				fd = await promisify(fs.open)(path, "a")
			catch err
				console.error "Failed to open logfile #{path}: #{err}"
				res.writeHead 500
				res.end JSON.stringify {error: "Opening logfile failed"}
				return
			try
				# TODO: Can we have a race condition here?"
				await promisify(fs.write) fd, data + "\n"
			catch err
				console.error "Writing to logfile #{path} failed: #{err}"
				res.writeHead 500
				res.end JSON.stringify {error: "Failed to write logfile"}
			finally
				fs.close fd, ->
			res.writeHead 200
			res.end()

	server.listen 8082
