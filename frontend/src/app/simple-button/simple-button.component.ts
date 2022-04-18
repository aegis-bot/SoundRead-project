import { Injectable, Input, Output, EventEmitter } from '@angular/core';
import { HttpClient, HttpHandler, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

import { Component, OnInit } from '@angular/core';
import { HttpHandlingService } from '../http-handling.service';

@Component({
  selector: 'app-simple-button',
  templateUrl: './simple-button.component.html',
  styleUrls: ['./simple-button.component.css']
})

/**
 * A husk button (only aesthetics).
 */
export class SimpleButtonComponent implements OnInit {
  @Input() displayText = "";
  @Input() buttonClass= "btn btn-primary btn-lg";
  @Input() isEnabled: boolean = true;
  constructor() { }
  ngOnInit(): void {
  }  
}
