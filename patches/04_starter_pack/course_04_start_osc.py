#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:13:08 2023

@author: esling
"""
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client

# OSC decorator
def osc_parse(func):
    '''decorates a python function to automatically transform args and kwargs coming from Max'''
    def func_embedding(address, *args):
        t_args = tuple(); kwargs = {}
        for a in args:
            if issubclass(type(a), str):
                if "=" in a:
                    key, value = a.split("=")
                    kwargs[key] = value
                else:
                    t_args = t_args + (a,)
            else:
                t_args = t_args + (a,)
        return func(*t_args, **kwargs)
    return func_embedding

class OSCServer():

    def __init__(self, in_port, out_port, ip="127.0.0.1", *args):
        self.ip = ip
        self.port_in = in_port
        self.port_out = out_port
        # Client object to use in functions
        self.client = udp_client.SimpleUDPClient(self.ip, self.port_out)
        # Create a dispatcher
        self.dispatch = dispatcher.Dispatcher()
        # Initialize all bindings
        self.init_bindings()
        # Create the server
        self.server = osc_server.BlockingOSCUDPServer((ip, self.port_in), self.dispatch)

    def init_bindings(self):
        # Mapping message to the function
        self.dispatch.map("/synth/frequency", self.synth_frequency)
        self.dispatch.map("/stop", self.stop_server)

    # Function to react to an incoming /synth/frequency message
    def synth_frequency(self, address, freq):
        print(f"Received frequency : {freq}")
        # Advanced complicated computing here
        new_freq = freq * 2
        # Send back the frequency
        self.client.send_message("/synth/frequency", new_freq)

    def start_server(self):
        # Launch the server
        self.server.serve_forever()

    def stop_server(self, *args):
        self.client.send_message("/terminated", "bang")
        self.server.shutdown()
        self.server.socket.close()