import { Injectable, Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { HttpClient, HttpHandler, HttpParams } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

import { HttpHandlingService } from '../http-handling.service';

@Component({
  selector: 'app-card-component',
  templateUrl: './card-component.component.html',
  styleUrls: ['./card-component.component.css']
})
export class CardComponentComponent implements OnInit {
  //no parameters unless got multiple uploads
  @Input() header = "";
  @Input() buttonDesc  = "";
  public  fileName: any;
  private file: File;

  @Output() receivedFileEvent = new EventEmitter<File>();

  constructor(private dataService: HttpHandlingService) { }

  ngOnInit(): void {
  }

  public onFileSelected($event: any) {
    console.log("onFileSelected");
    this.file = (<HTMLInputElement>event.target).files[0];
    if(this.file) {
      this.fileName = this.file.name
    }
    this.receivedFileEvent.emit(this.file);
  }
}
