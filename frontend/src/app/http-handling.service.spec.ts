import { TestBed } from '@angular/core/testing';

import { HttpHandlingService } from './http-handling.service';

describe('HttpHandlingServiceService', () => {
  let service: HttpHandlingService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(HttpHandlingService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
