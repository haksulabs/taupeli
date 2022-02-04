{WebSocketServer} = require 'ws'
uuid = require 'uuid'
fs = require 'fs'
{promisify} = require 'util'

# TODO:
# SSL
# Configurable port
# More session info from request?

isValidName = (name) ->
	return /^[ \w_-]*$/.test name

do ->
	server = new WebSocketServer {port: 8082}
	
	server.on "connection", (ws, request) ->
		session_id = uuid.v1()
		url = new URL request.url, "ws://example.com"
		session_name = url.searchParams.get("name") ? ""
		if not isValidName session_id
			ws.close(1011, JSON.stringify {error: "Invalid session name"})
			return
		
		session_id = session_name + "-" + session_id
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
